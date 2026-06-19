import Foundation

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
    static func scrape(
        backend: SyntheticBackend,
        baseURL: String,
        apiKey: String?
    ) async throws -> [String] {
        if backend == .mlx { return [] }

        let resolvedURL = effectiveBaseURL(backend: backend, baseURL: baseURL)
        guard let endpoint = scrapeEndpoint(backend: backend, baseURL: resolvedURL) else {
            throw ScrapeError.invalidBaseURL
        }
        let trimmedKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let usesAuth = shouldAttachAuth(backend: backend, hasKey: !trimmedKey.isEmpty)

        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
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
    private static func scrapeEndpoint(backend: SyntheticBackend, baseURL: String) -> URL? {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let parsed = URL(string: trimmed) else { return nil }
        let scheme = parsed.scheme?.lowercased() ?? ""
        guard scheme == "http" || scheme == "https" else { return nil }

        if backend == .ollama {
            return ollamaTagsURL(from: parsed)
        }
        return openAICompatibleModelsURL(from: parsed)
    }

    private static func ollamaTagsURL(from base: URL) -> URL? {
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
