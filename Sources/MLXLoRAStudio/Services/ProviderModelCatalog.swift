import Foundation

/// Live scrapes the list of available model ids from a synthetic-data
/// provider's endpoint, so the UI can show a model picker populated
/// with whatever the provider actually has right now instead of a
/// hand-curated "suggested" list.
///
/// The "on-the-fly" / rescrape flow is what powers the
/// `SyntheticProviderModelPicker`. The picker auto-scrapes when it
/// first appears and again whenever the user switches to a
/// different backend, and a refresh button next to the picker
/// triggers a manual rescrape for the rare case where a new model
/// shows up mid-run.
///
/// We hit a small set of well-known endpoints. All the cloud
/// providers we support (OpenAI, OpenRouter, oMLX, LM Studio,
/// Custom) speak the OpenAI-compatible `GET {baseURL}/models`
/// schema, which returns `{"data": [{"id": "…"}, …]}`. Ollama
/// deliberately does NOT speak that schema at the OpenAI shim, so
/// we hit its native `GET {baseURL}/../api/tags` endpoint (which
/// returns `{"models": [{"name": "…"}, …]}`) instead. The base URL
/// for Ollama in the app is `http://localhost:11434/v1`, so the
/// native path is `http://localhost:11434/api/tags` — the scraper
/// normalises the trailing `/v1`.
///
/// MLX (the local-path backend) has no remote catalog, so
/// `scrape(backend:baseURL:apiKey:)` returns an empty list for
/// that case and the UI falls back to letting the user type a
/// model id freely.
enum ProviderModelCatalog {
    enum ScrapeError: LocalizedError {
        case invalidBaseURL
        case requestFailed(underlying: Error)
        case httpStatus(Int)
        case malformedResponse

        var errorDescription: String? {
            switch self {
            case .invalidBaseURL:
                return "Provider URL is empty or not a valid http(s) URL."
            case .requestFailed(let underlying):
                return "Could not reach the provider: \(underlying.localizedDescription)"
            case .httpStatus(let code):
                return "Provider returned HTTP \(code)."
            case .malformedResponse:
                return "Provider returned a response the scraper could not parse."
            }
        }
    }

    /// Fetch the live model id list for the given provider.
    ///
    /// - Parameters:
    ///   - backend: the synthetic-data provider. MLX returns an
    ///     empty list (no remote catalog). Ollama uses its native
    ///     `/api/tags` endpoint. All others use the OpenAI-
    ///     compatible `/v1/models` endpoint.
    ///   - baseURL: the user-edited base URL for the provider. The
    ///     scraper falls back to `backend.defaultBaseURL` if this
    ///     is empty.
    ///   - apiKey: optional bearer token. Cloud providers (OpenAI,
    ///     OpenRouter, oMLX) get it; local servers (Ollama, LM
    ///     Studio) do NOT — the absence of a token is itself a
    ///     signal that the user is hitting a local server.
    static func scrape(
        backend: SyntheticBackend,
        baseURL: String,
        apiKey: String?
    ) async throws -> [String] {
        // MLX is purely local — the model is whatever path the user
        // types into the `model` field. We return an empty list so
        // the picker's menu only shows the user's typed value.
        if backend == .mlx { return [] }

        let resolvedURL = effectiveBaseURL(backend: backend, baseURL: baseURL)
        guard let endpoint = scrapeEndpoint(backend: backend, baseURL: resolvedURL) else {
            throw ScrapeError.invalidBaseURL
        }
        let trimmedKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let usesAuth = shouldAttachAuth(backend: backend, hasKey: !trimmedKey.isEmpty)

        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        // Generous timeout — local servers can be slow on cold
        // start, and a stuck spinner is a worse experience than a
        // small delay. The picker shows a progress indicator the
        // whole time, so the user is never blocked.
        request.timeoutInterval = 15
        if usesAuth {
            request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw ScrapeError.requestFailed(underlying: error)
        }
        guard let http = response as? HTTPURLResponse else {
            throw ScrapeError.malformedResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw ScrapeError.httpStatus(http.statusCode)
        }
        return try parseModelIDs(backend: backend, data: data)
    }

    // MARK: - URL helpers

    /// Pick the URL the scraper will hit. OpenAI-compatible backends
    /// get `{baseURL}/models`; Ollama gets the native
    /// `…/api/tags` endpoint. Returns nil when the base URL is not
    /// a parseable http(s) URL.
    private static func scrapeEndpoint(backend: SyntheticBackend, baseURL: String) -> URL? {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let parsed = URL(string: trimmed) else { return nil }
        let scheme = parsed.scheme?.lowercased() ?? ""
        guard scheme == "http" || scheme == "https" else { return nil }

        if backend == .ollama {
            // Ollama's default base URL in the app is
            // `http://localhost:11434/v1`. Strip the `/v1` so we
            // hit its native tag list (`/api/tags`) which returns
            // the full set of models the user has pulled locally.
            return ollamaTagsURL(from: parsed)
        }
        return openAICompatibleModelsURL(from: parsed)
    }

    private static func ollamaTagsURL(from base: URL) -> URL? {
        // Use URLComponents so we don't accidentally drop the host
        // or port while rewriting the path.
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            return nil
        }
        var path = components.path
        if path.hasSuffix("/v1") {
            path = String(path.dropLast("/v1".count))
        } else if path.hasSuffix("/v1/") {
            path = String(path.dropLast("/v1/".count))
        }
        if path.hasSuffix("/") { path = String(path.dropLast()) }
        if !path.hasSuffix("/api/tags") {
            if !path.isEmpty && !path.hasSuffix("/") { path += "/" }
            path += "api/tags"
        }
        components.path = path
        return components.url
    }

    private static func openAICompatibleModelsURL(from base: URL) -> URL? {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            return nil
        }
        var path = components.path
        // Strip a trailing slash for consistent concatenation.
        if path.hasSuffix("/") { path = String(path.dropLast()) }
        if !path.hasSuffix("/models") {
            if !path.isEmpty { path += "/" }
            path += "models"
        }
        components.path = path
        return components.url
    }

    private static func effectiveBaseURL(backend: SyntheticBackend, baseURL: String) -> String {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? backend.defaultBaseURL : trimmed
    }

    // MARK: - Auth policy

    /// Cloud providers always want the bearer token when one is
    /// set. Local servers (Ollama, LM Studio) deliberately do NOT
    /// receive one — sending an `Authorization` header to a server
    /// that doesn't expect it has been known to make some local
    /// servers return 401s. The picker should only ever scrape
    /// these with no auth header.
    private static func shouldAttachAuth(backend: SyntheticBackend, hasKey: Bool) -> Bool {
        guard hasKey else { return false }
        switch backend {
        case .ollama, .lmstudio:
            return false
        case .mlx, .openai, .openrouter, .omlx, .custom:
            return true
        }
    }

    // MARK: - Response parsers

    /// Try every schema the provider might return. OpenAI-compatible
    /// endpoints use `{"data": [{"id": "…"}]}`; Ollama uses
    /// `{"models": [{"name": "…"}]}`; a few providers (and OpenAI
    /// itself for fine-tunes) occasionally use a flat array. We
    /// keep the parsing tolerant so a future provider that returns
    /// one of these shapes "just works" without a code change.
    private static func parseModelIDs(backend: SyntheticBackend, data: Data) throws -> [String] {
        if let openAI = try? decodeOpenAIShape(from: data) { return openAI }
        if let ollama = try? decodeOllamaShape(from: data) { return ollama }
        if let flat = try? decodeFlatShape(from: data) { return flat }
        throw ScrapeError.malformedResponse
    }

    private struct OpenAIResponse: Decodable {
        let data: [Entry]?
        struct Entry: Decodable { let id: String? }
    }

    private struct OllamaResponse: Decodable {
        let models: [Entry]?
        struct Entry: Decodable { let name: String? }
    }

    private struct FlatResponse: Decodable {
        let id: String?
        let name: String?
    }

    private static func decodeOpenAIShape(from data: Data) throws -> [String] {
        let decoded = try JSONDecoder().decode(OpenAIResponse.self, from: data)
        let ids = (decoded.data ?? []).compactMap { $0.id?.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !ids.isEmpty else { throw ScrapeError.malformedResponse }
        return deduped(ids)
    }

    private static func decodeOllamaShape(from data: Data) throws -> [String] {
        let decoded = try JSONDecoder().decode(OllamaResponse.self, from: data)
        let ids = (decoded.models ?? []).compactMap { $0.name?.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !ids.isEmpty else { throw ScrapeError.malformedResponse }
        return deduped(ids)
    }

    private static func decodeFlatShape(from data: Data) throws -> [String] {
        let decoded = try JSONDecoder().decode([FlatResponse].self, from: data)
        let ids = decoded.compactMap { entry -> String? in
            let candidate = entry.id ?? entry.name
            return candidate?.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !ids.isEmpty else { throw ScrapeError.malformedResponse }
        return deduped(ids)
    }

    /// Drop duplicates while preserving the order the server gave
    /// us. Most providers already return a sorted list, but a few
    /// (notably Ollama) return models in arbitrary order — sorting
    /// alphabetically here gives a stable, scannable dropdown.
    private static func deduped(_ ids: [String]) -> [String] {
        var seen = Set<String>()
        var unique: [String] = []
        for id in ids {
            guard !id.isEmpty, seen.insert(id).inserted else { continue }
            unique.append(id)
        }
        return unique.sorted()
    }
}
