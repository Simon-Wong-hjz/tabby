import { EventEmitter } from "events";
import path from "path";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import { deepmerge } from "deepmerge-ts";
import { getProperty, setProperty, deleteProperty } from "dot-prop";
import createClient from "openapi-fetch";
import type { ParseAs } from "openapi-fetch";
import * as semver from "semver";
import type { paths as TabbyApi, components as TabbyApiComponents } from "./types/tabbyApi";
import type {
  Agent,
  AgentStatus,
  AgentIssue,
  AgentEvent,
  ClientProperties,
  AgentInitOptions,
  AbortSignalOption,
  ServerHealthState,
  CompletionRequest,
  CompletionResponse,
  LogEventRequest,
} from "./Agent";
import { dataStore as defaultDataStore, DataStore } from "./dataStore";
import {
  isBlank,
  abortSignalFromAnyOf,
  HttpError,
  isTimeoutError,
  isCanceledError,
  isUnauthorizedError,
  errorToString,
} from "./utils";
import { Auth } from "./Auth";
import { AgentConfig, PartialAgentConfig, defaultAgentConfig } from "./AgentConfig";
import { configFile } from "./configFile";
import { CompletionCache } from "./CompletionCache";
import { CompletionDebounce } from "./CompletionDebounce";
import { CompletionContext } from "./CompletionContext";
import { preCacheProcess, postCacheProcess } from "./postprocess";
import { getLogger, logDestinations, fileLogger } from "./logger";
import { AnonymousUsageLogger } from "./AnonymousUsageLogger";
import { CompletionProviderStats, CompletionProviderStatsEntry } from "./CompletionProviderStats";
import { loadTlsCaCerts } from "./loadCaCerts";

export class TabbyAgent extends EventEmitter implements Agent {
  private readonly logger = getLogger("TabbyAgent");
  private anonymousUsageLogger = new AnonymousUsageLogger();
  private config: AgentConfig = defaultAgentConfig;
  private userConfig: PartialAgentConfig = {}; // config from `~/.tabby-client/agent/config.toml`
  private clientConfig: PartialAgentConfig = {}; // config from `initialize` and `updateConfig` method
  private serverProvidedConfig: PartialAgentConfig = {}; // config fetched from server and saved in dataStore
  private status: AgentStatus = "notInitialized";
  private issues: AgentIssue["name"][] = [];
  private serverHealthState?: ServerHealthState;
  private connectionErrorMessage?: string;
  private dataStore?: DataStore;
  private api?: ReturnType<typeof createClient<TabbyApi>>;
  private auth?: Auth;
  private completionCache = new CompletionCache();
  private completionDebounce = new CompletionDebounce();
  private completionMutexAbortController?: AbortController;
  private completionProviderStats = new CompletionProviderStats();
  static readonly tryConnectInterval = 1000 * 30; // 30s
  private tryingConnectTimer: ReturnType<typeof setInterval>;
  static readonly submitStatsInterval = 1000 * 60 * 60 * 24; // 24h
  private submitStatsTimer: ReturnType<typeof setInterval>;

  constructor() {
    super();

    this.tryingConnectTimer = setInterval(async () => {
      if (this.status === "disconnected") {
        this.logger.debug("Trying to connect...");
        await this.healthCheck();
      }
    }, TabbyAgent.tryConnectInterval);

    this.submitStatsTimer = setInterval(async () => {
      await this.submitStats();
    }, TabbyAgent.submitStatsInterval);
  }

  private async applyConfig() {
    this.logger.info("Applying updated config...");
    const oldConfig = this.config;
    const oldStatus = this.status;

    this.config = deepmerge(
      defaultAgentConfig,
      this.userConfig,
      this.clientConfig,
      this.serverProvidedConfig,
    ) as AgentConfig;
    this.logger.trace("Updated config:", this.config);

    if (fileLogger) {
      fileLogger.level = this.config.logs.level;
    }
    this.anonymousUsageLogger.disabled = this.config.anonymousUsageTracking.disable;

    await loadTlsCaCerts(this.config.tls);

    if (isBlank(this.config.server.token) && this.config.server.requestHeaders["Authorization"] === undefined) {
      if (this.config.server.endpoint !== this.auth?.endpoint) {
        this.auth = new Auth(this.config.server.endpoint);
        await this.auth.init({ dataStore: this.dataStore });
        this.auth.on("updated", () => {
          this.setupApi();
        });
      }
    } else {
      // If auth token is provided, use it directly.
      this.auth = undefined;
    }

    // If server config changed, clear server related state
    if (!deepEqual(oldConfig.server, this.config.server)) {
      this.serverHealthState = undefined;
      this.completionProviderStats.resetWindowed();
      this.popIssue("slowCompletionResponseTime");
      this.popIssue("highCompletionTimeoutRate");
      this.popIssue("connectionFailed");
      this.connectionErrorMessage = undefined;
    }

    if (!this.api || !deepEqual(oldConfig.server, this.config.server)) {
      await this.setupApi();
    }

    this.logger.info("Completed applying updated config.");
    const event: AgentEvent = { event: "configUpdated", config: this.config };
    super.emit("configUpdated", event);
    if (
      !deepEqual(oldConfig.server, this.config.server) &&
      oldStatus === "unauthorized" &&
      this.status === "unauthorized"
    ) {
      // If server config changed and status remain `unauthorized`, we want to emit `authRequired` again.
      // but `changeStatus` will not emit `authRequired` if status is not changed, so we emit it manually here.
      this.emitAuthRequired();
    }
  }

  private async setupApi() {
    const auth = !isBlank(this.config.server.token)
      ? `Bearer ${this.config.server.token}`
      : this.auth?.token
        ? `Bearer ${this.auth.token}`
        : undefined;
    this.api = createClient<TabbyApi>({
      baseUrl: this.config.server.endpoint.replace(/\/+$/, ""), // remove trailing slash
      headers: {
        Authorization: auth,
        ...this.config.server.requestHeaders,
      },
    });
    await this.healthCheck();
  }

  private changeStatus(status: AgentStatus) {
    if (this.status != status) {
      this.status = status;
      this.logger.info(`Status changed: ${status}.`);
      const event: AgentEvent = { event: "statusChanged", status };
      super.emit("statusChanged", event);
      if (this.status === "unauthorized") {
        this.emitAuthRequired();
      }
    }
  }

  private issueFromName(issueName: AgentIssue["name"]): AgentIssue {
    switch (issueName) {
      case "highCompletionTimeoutRate":
        return {
          name: "highCompletionTimeoutRate",
          completionResponseStats: this.completionProviderStats.windowed().stats,
        };
      case "slowCompletionResponseTime":
        return {
          name: "slowCompletionResponseTime",
          completionResponseStats: this.completionProviderStats.windowed().stats,
        };
      case "connectionFailed":
        return {
          name: "connectionFailed",
          message: this.connectionErrorMessage,
        };
    }
  }

  private pushIssue(issue: AgentIssue["name"]) {
    if (!this.issues.includes(issue)) {
      this.issues.push(issue);
      this.logger.info(`Issue created: ${issue}.`);
      this.emitIssueUpdated();
    }
  }

  private popIssue(issue: AgentIssue["name"]) {
    const index = this.issues.indexOf(issue);
    if (index >= 0) {
      this.issues.splice(index, 1);
      this.logger.info(`Issue removed: ${issue}.`);
      this.emitIssueUpdated();
    }
  }

  private emitAuthRequired() {
    const event: AgentEvent = { event: "authRequired", server: this.config.server };
    super.emit("authRequired", event);
  }

  private emitIssueUpdated() {
    const event: AgentEvent = { event: "issuesUpdated", issues: this.issues };
    super.emit("issuesUpdated", event);
  }

  private async submitStats() {
    const stats = this.completionProviderStats.stats();
    if (stats.completion_request.count > 0) {
      await this.anonymousUsageLogger.event("AgentStats", { stats });
      this.completionProviderStats.reset();
    }
  }

  private createAbortSignal(options?: { signal?: AbortSignal; timeout?: number }): AbortSignal {
    const timeout = Math.min(0x7fffffff, options?.timeout || this.config.server.requestTimeout);
    return abortSignalFromAnyOf([AbortSignal.timeout(timeout), options?.signal]);
  }

  private async healthCheck(options?: { signal?: AbortSignal; method?: "GET" | "POST" }): Promise<void> {
    const requestId = uuid();
    const requestPath = "/v1/health";
    const requestDescription = `${options?.method || "GET"} ${this.config.server.endpoint + requestPath}`;
    const requestOptions = {
      signal: this.createAbortSignal({ signal: options?.signal }),
    };
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Health check request: ${requestDescription}. [${requestId}]`);
      let response;
      if (options?.method === "POST") {
        response = await this.api.POST(requestPath, requestOptions);
      } else {
        response = await this.api.GET(requestPath, requestOptions);
      }
      this.logger.debug(`Health check response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Health check response data: [${requestId}]`, response.data);
      const healthState = response.data;
      this.popIssue("connectionFailed");
      this.connectionErrorMessage = undefined;
      if (
        typeof healthState === "object" &&
        healthState["model"] !== undefined &&
        healthState["device"] !== undefined
      ) {
        this.serverHealthState = healthState;
        this.anonymousUsageLogger.uniqueEvent("AgentConnected", healthState);

        // schedule fetch server config later, no await
        this.fetchServerProvidedConfig();
      }
      this.changeStatus("ready");
    } catch (error) {
      this.serverHealthState = undefined;
      if (error instanceof HttpError && error.status == 405 && options?.method !== "POST") {
        return await this.healthCheck({ method: "POST" });
      } else if (isUnauthorizedError(error)) {
        this.changeStatus("unauthorized");
      } else {
        if (isCanceledError(error)) {
          this.logger.debug(`Health check request canceled. [${requestId}]`);
          this.connectionErrorMessage = `${requestDescription} canceled.`;
        } else if (isTimeoutError(error)) {
          this.logger.error(`Health check request timed out. [${requestId}]`, error);
          this.connectionErrorMessage = `${requestDescription} timed out.`;
        } else {
          this.logger.error(`Health check request failed. [${requestId}]`, error);
          const message = error instanceof Error ? errorToString(error) : JSON.stringify(error);
          this.connectionErrorMessage = `${requestDescription} failed: \n${message}`;
        }
        this.pushIssue("connectionFailed");
        this.changeStatus("disconnected");
      }
    }
  }

  private async fetchServerProvidedConfig(): Promise<void> {
    const serverVersion = semver.coerce(this.serverHealthState?.version.git_describe);
    if (serverVersion && semver.lt(serverVersion, "0.9.0")) {
      this.logger.debug(`Skip fetching server provided config due to server version: ${serverVersion}.`);
      return;
    }
    const requestId = uuid();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = "/v1beta/server_setting";
      const requestDescription = `GET ${this.config.server.endpoint + requestPath}`;
      this.logger.debug(`Fetch server provided config request: ${requestDescription}. [${requestId}]`);
      const response = await this.api.GET(requestPath);
      this.logger.debug(`Fetch server provided config response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Fetch server provided config response data: [${requestId}]`, response.data);
      const fetchedConfig = response.data;
      const serverProvidedConfig: PartialAgentConfig = {};
      if (fetchedConfig.disable_client_side_telemetry) {
        serverProvidedConfig.anonymousUsageTracking = {
          disable: true,
        };
      }

      if (!deepEqual(serverProvidedConfig, this.serverProvidedConfig)) {
        this.serverProvidedConfig = serverProvidedConfig;
        await this.applyConfig();
        if (this.dataStore) {
          if (!this.dataStore.data.serverConfig) {
            this.dataStore.data.serverConfig = {};
          }
          this.dataStore.data.serverConfig[this.config.server.endpoint] = this.serverProvidedConfig;
          try {
            await this.dataStore.save();
          } catch (error) {
            this.logger.error("Failed to save server provided config.", error);
          }
        }
      }
    } catch (error) {
      if (isUnauthorizedError(error)) {
        this.logger.debug(`Fetch server provided config request failed due to unauthorized. [${requestId}]`);
      } else if (error! instanceof HttpError) {
        this.logger.error(`Fetch server provided config request failed. [${requestId}]`, error);
      }
    }
  }

  private async createSegments(context: CompletionContext): Promise<TabbyApiComponents["schemas"]["Segments"] | null> {
    // max lines in prefix and suffix configurable
    const maxPrefixLines = this.config.completion.prompt.maxPrefixLines;
    const maxSuffixLines = this.config.completion.prompt.maxSuffixLines;
    const { prefixLines, suffixLines } = context;
    const prefix = prefixLines.slice(Math.max(prefixLines.length - maxPrefixLines, 0)).join("");
    const suffix = suffixLines.slice(0, maxSuffixLines).join("");
    if (isBlank(prefix)) {
      return null;
    }

    // filepath && git_url
    let relativeFilepathRoot: string | undefined = undefined;
    let filepath: string | undefined = undefined;
    let gitUrl: string | undefined = undefined;
    const { workspace, git } = context;
    if (git && git.remotes.length > 0) {
      // find remote url: origin > upstream > first
      const remote =
        git.remotes.find((remote) => remote.name === "origin") ||
        git.remotes.find((remote) => remote.name === "upstream") ||
        git.remotes[0];
      if (remote) {
        relativeFilepathRoot = git.root;
        gitUrl = remote.url;
      }
    }
    // if relativeFilepathRoot is not set by git context, use path relative to workspace
    if (!relativeFilepathRoot && workspace) {
      relativeFilepathRoot = workspace;
    }
    if (relativeFilepathRoot) {
      filepath = path.relative(relativeFilepathRoot, context.filepath);
    }

    // declarations
    const declarations = context.declarations?.map((declaration) => {
      let declarationFilepath = declaration.filepath;
      if (relativeFilepathRoot && declarationFilepath.startsWith(relativeFilepathRoot)) {
        declarationFilepath = path.relative(relativeFilepathRoot, declarationFilepath);
      }
      return {
        filepath: declarationFilepath,
        body: declaration.text,
      };
    });

    // snippets
    const relevantSnippetsFromChangedFiles = context.relevantSnippetsFromChangedFiles
      // deduplicate
      ?.filter(
        (snippet) =>
          // Remove snippet if find a declaration from the same file and range is overlapping
          !context.declarations?.find((declaration) => {
            return (
              declaration.filepath === snippet.filepath &&
              // Is range overlapping
              Math.max(declaration.offset, snippet.offset) <=
                Math.min(declaration.offset + declaration.text.length, snippet.offset + snippet.text.length)
            );
          }),
      )
      .map((snippet) => {
        let snippetFilepath = snippet.filepath;
        if (relativeFilepathRoot && snippetFilepath.startsWith(relativeFilepathRoot)) {
          snippetFilepath = path.relative(relativeFilepathRoot, snippetFilepath);
        }
        return {
          filepath: snippetFilepath,
          body: snippet.text,
          score: snippet.score,
        };
      })
      .sort((a, b) => b.score - a.score);

    // clipboard
    let clipboard = undefined;
    const clipboardConfig = this.config.completion.prompt.clipboard;
    if (context.clipboard.length >= clipboardConfig.minChars && context.clipboard.length <= clipboardConfig.maxChars) {
      clipboard = context.clipboard;
    }
    return {
      prefix,
      suffix,
      filepath,
      git_url: gitUrl,
      declarations,
      relevant_snippets_from_changed_files: relevantSnippetsFromChangedFiles,
      clipboard,
    };
  }

  public async initialize(options: AgentInitOptions): Promise<boolean> {
    this.logger.info("Initializing...");
    this.logger.trace("Initialization options:", options);
    if (options.loggers) {
      logDestinations.attach(...options.loggers);
    }
    this.dataStore = options.dataStore ?? defaultDataStore;
    if (this.dataStore) {
      try {
        await this.dataStore.load();
        if ("watch" in this.dataStore && typeof this.dataStore.watch === "function") {
          this.dataStore.watch();
        }
      } catch (error) {
        this.logger.error("Failed to load stored data.", error);
      }
    }
    await this.anonymousUsageLogger.init({ dataStore: this.dataStore });
    if (options.clientProperties) {
      if (options.clientProperties.session) {
        Object.entries(options.clientProperties.session).forEach(([key, value]) => {
          this.anonymousUsageLogger.setSessionProperties(key, value);
        });
      }
      if (options.clientProperties.user) {
        Object.entries(options.clientProperties.user).forEach(([key, value]) => {
          this.anonymousUsageLogger.setUserProperties(key, value);
        });
      }
    }
    if (configFile) {
      await configFile.load();
      this.userConfig = configFile.config;
      configFile.on("updated", async (config) => {
        this.userConfig = config;
        await this.applyConfig();
      });
      configFile.watch();
    }
    if (options.config) {
      this.clientConfig = options.config;
    }
    if (this.dataStore) {
      const localConfig = deepmerge(defaultAgentConfig, this.userConfig, this.clientConfig) as AgentConfig;
      this.serverProvidedConfig = this.dataStore?.data.serverConfig?.[localConfig.server.endpoint] ?? {};
      if (this.dataStore instanceof EventEmitter) {
        this.dataStore.on("updated", async () => {
          const localConfig = deepmerge(defaultAgentConfig, this.userConfig, this.clientConfig) as AgentConfig;
          const storedServerConfig = defaultDataStore?.data.serverConfig?.[localConfig.server.endpoint];
          if (!deepEqual(storedServerConfig, this.serverProvidedConfig)) {
            this.serverProvidedConfig = storedServerConfig ?? {};
            await this.applyConfig();
          }
        });
      }
    }
    await this.applyConfig();
    await this.anonymousUsageLogger.uniqueEvent("AgentInitialized");
    this.logger.info("Initialized.");
    return this.status !== "notInitialized";
  }

  public async finalize(): Promise<boolean> {
    if (this.status === "finalized") {
      return false;
    }
    this.logger.info(`Finalizing...`);

    await this.submitStats();

    if (this.tryingConnectTimer) {
      clearInterval(this.tryingConnectTimer);
    }
    if (this.submitStatsTimer) {
      clearInterval(this.submitStatsTimer);
    }
    this.changeStatus("finalized");
    this.logger.info(`Finalized.`);
    return true;
  }

  public async updateClientProperties(type: keyof ClientProperties, key: string, value: any): Promise<boolean> {
    this.logger.trace(`Client properties updated.`, { type, key, value });
    switch (type) {
      case "session":
        this.anonymousUsageLogger.setSessionProperties(key, value);
        break;
      case "user":
        this.anonymousUsageLogger.setUserProperties(key, value);
        break;
    }
    return true;
  }

  public async updateConfig(key: string, value: any): Promise<boolean> {
    this.logger.trace(`Config updated.`, { key, value });
    const current = getProperty(this.clientConfig, key);
    if (!deepEqual(current, value)) {
      if (value === undefined) {
        deleteProperty(this.clientConfig, key);
      } else {
        setProperty(this.clientConfig, key, value);
      }
      await this.applyConfig();
    }
    return true;
  }

  public async clearConfig(key: string): Promise<boolean> {
    return await this.updateConfig(key, undefined);
  }

  public getConfig(): AgentConfig {
    return this.config;
  }

  public getStatus(): AgentStatus {
    return this.status;
  }

  public getIssues(): AgentIssue["name"][] {
    return this.issues;
  }

  public getIssueDetail<T extends AgentIssue>(options: { index?: number; name?: T["name"] }): T | null {
    const issues = this.getIssues();
    if (options.index !== undefined && options.index < issues.length) {
      return this.issueFromName(issues[options.index]!) as T;
    } else if (options.name !== undefined && this.issues.includes(options.name)) {
      return this.issueFromName(options.name) as T;
    } else {
      return null;
    }
  }

  public getServerHealthState(): ServerHealthState | null {
    return this.serverHealthState ?? null;
  }

  // @deprecated Tabby Cloud auth
  public async requestAuthUrl(options?: AbortSignalOption): Promise<{ authUrl: string; code: string } | null> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    await this.healthCheck(options);
    if (this.status !== "unauthorized" || !this.auth) {
      return null;
    } else {
      return await this.auth.requestAuthUrl(options);
    }
  }

  // @deprecated Tabby Cloud auth
  public async waitForAuthToken(code: string, options?: AbortSignalOption): Promise<void> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    if (this.status !== "unauthorized" || !this.auth) {
      return;
    }
    await this.auth.pollingToken(code, options);
    await this.setupApi();
  }

  public async provideCompletions(
    request: CompletionRequest,
    options?: AbortSignalOption,
  ): Promise<CompletionResponse> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    this.logger.debug("Function providedCompletions called.");
    this.logger.trace("Completion request:", request);
    if (this.completionMutexAbortController) {
      const reason = new Error("Aborted due to new request.");
      reason.name = "AbortError";
      this.completionMutexAbortController.abort(reason);
    }
    this.completionMutexAbortController = new AbortController();
    const signal = abortSignalFromAnyOf([this.completionMutexAbortController.signal, options?.signal]);

    let completionResponse: CompletionResponse;
    let stats: CompletionProviderStatsEntry | undefined = {
      triggerMode: request.manually ? "manual" : "auto",
      cacheHit: false,
      aborted: false,
      requestSent: false,
      requestLatency: 0,
      requestCanceled: false,
      requestTimeout: false,
    };
    let requestStartedAt: number | undefined;

    const context = new CompletionContext(request);
    try {
      if (this.completionCache.has(context)) {
        // Cache hit
        stats.cacheHit = true;
        // Debounce before returning cached response
        await this.completionDebounce.debounce(
          {
            request,
            config: this.config.completion.debounce,
            responseTime: 0,
          },
          { signal },
        );

        this.logger.info("Completion cache hit.");
        completionResponse = this.completionCache.get(context)!;
      } else {
        // Cache miss
        stats.cacheHit = false;
        const segments = await this.createSegments(context);
        if (!segments) {
          stats = undefined; // no need to record stats when no segments
          this.logger.info("No context segment available.");
          completionResponse = {
            id: "agent-" + uuid(),
            choices: [],
          };
        } else {
          // Debounce before sending request
          await this.completionDebounce.debounce(
            {
              request,
              config: this.config.completion.debounce,
              responseTime: this.completionProviderStats.windowed().stats.averageResponseTime,
            },
            options,
          );

          // Send http request
          this.logger.info(`Fetching completions...`);
          const requestId = uuid();
          stats.requestSent = true;
          requestStartedAt = performance.now();
          try {
            if (!this.api) {
              throw new Error("http client not initialized");
            }
            const requestPath = "/v1/completions";
            const requestOptions = {
              body: {
                language: request.language,
                segments,
                user: this.auth?.user,
              },
              signal: this.createAbortSignal({ signal }),
            };
            const requestDescription = `POST ${this.config.server.endpoint + requestPath}`;
            this.logger.debug(`Completion request: ${requestDescription}. [${requestId}]`);
            this.logger.trace(`Completion request body: [${requestId}]`, requestOptions.body);
            const response = await this.api.POST(requestPath, requestOptions);
            this.logger.debug(`Completion response status: ${response.response.status}. [${requestId}]`);
            if (response.error || !response.response.ok) {
              throw new HttpError(response.response);
            }
            this.logger.trace(`Completion response data: [${requestId}]`, response.data);
            const responseData = response.data;
            stats.requestLatency = performance.now() - requestStartedAt;
            completionResponse = {
              id: responseData.id,
              choices: responseData.choices.map((choice) => {
                return {
                  index: choice.index,
                  text: choice.text,
                  replaceRange: {
                    start: request.position,
                    end: request.position,
                  },
                };
              }),
            };
          } catch (error) {
            if (isCanceledError(error)) {
              this.logger.info(`Fetching completions canceled.`);
              this.logger.debug(`Completion request canceled. [${requestId}]`);
              stats.requestCanceled = true;
              stats.requestLatency = performance.now() - requestStartedAt;
            } else {
              this.logger.info(`Fetching completions failed.`);
              if (isTimeoutError(error)) {
                this.logger.debug(`Completion request timed out. [${requestId}]`);
                stats.requestTimeout = true;
                stats.requestLatency = NaN;
              } else if (isUnauthorizedError(error)) {
                this.logger.debug(`Completion request failed due to unauthorized. [${requestId}]`);
                this.healthCheck(); // schedule a health check
              } else {
                this.logger.error(`Completion request failed. [${requestId}]`, error);
                this.healthCheck(); // schedule a health check
              }
            }
            // rethrow error
            throw error;
          }
          // Postprocess (pre-cache)
          completionResponse = await preCacheProcess(context, this.config.postprocess, completionResponse);
          if (signal.aborted) {
            throw signal.reason;
          }
          // Build cache
          this.completionCache.buildCache(context, JSON.parse(JSON.stringify(completionResponse)));
        }
      }
      // Postprocess (post-cache)
      completionResponse = await postCacheProcess(context, this.config.postprocess, completionResponse);
      if (signal.aborted) {
        throw signal.reason;
      }
    } catch (error) {
      if (isCanceledError(error) || isTimeoutError(error)) {
        if (stats) {
          stats.aborted = true;
        }
      } else {
        // unexpected error
        stats = undefined;
      }
      // rethrow error
      throw error;
    } finally {
      if (stats) {
        this.completionProviderStats.add(stats);

        if (stats.requestSent && !stats.requestCanceled) {
          const windowedStats = this.completionProviderStats.windowed();
          const checkResult = this.completionProviderStats.check(windowedStats);
          switch (checkResult) {
            case "healthy":
              this.popIssue("slowCompletionResponseTime");
              this.popIssue("highCompletionTimeoutRate");
              break;
            case "highTimeoutRate":
              this.popIssue("slowCompletionResponseTime");
              this.pushIssue("highCompletionTimeoutRate");
              break;
            case "slowResponseTime":
              this.popIssue("highCompletionTimeoutRate");
              this.pushIssue("slowCompletionResponseTime");
              break;
          }
        }
      }
    }
    this.logger.info(`Completed processing completions, choices returned: ${completionResponse.choices.length}.`);
    this.logger.trace("Completion response:", { context, completionResponse });
    return completionResponse;
  }

  public async postEvent(request: LogEventRequest, options?: AbortSignalOption): Promise<boolean> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    this.completionProviderStats.addEvent(request.type);
    const requestId = uuid();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = "/v1/events";
      const requestOptions = {
        body: request,
        params: {
          query: {
            select_kind: request.select_kind,
          },
        },
        signal: this.createAbortSignal(options),
        parseAs: "text" as ParseAs,
      };
      const requestDescription = `POST ${this.config.server.endpoint + requestPath}`;
      this.logger.debug(`Event request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Event request body: [${requestId}]`, requestOptions.body);
      const response = await this.api.POST(requestPath, requestOptions);
      this.logger.debug(`Event response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Event response data: [${requestId}]`, response.data);
      return true;
    } catch (error) {
      if (isUnauthorizedError(error)) {
        this.logger.debug(`Completion request failed due to unauthorized. [${requestId}]`);
      } else {
        this.logger.error(`Event request failed. [${requestId}]`, error);
      }
      this.healthCheck(); // schedule a health check
      return false;
    }
  }
}
