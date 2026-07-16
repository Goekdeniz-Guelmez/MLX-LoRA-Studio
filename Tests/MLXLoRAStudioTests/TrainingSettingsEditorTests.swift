import Testing
@testable import MLXLoRAStudio

struct TrainingSettingsEditorTests {
    @Test func appliesTypedChanges() throws {
        var config = TrainingConfig()
        try TrainingSettingsEditor.apply(.init(name: "learning_rate", value: "0.0002"), to: &config)
        try TrainingSettingsEditor.apply(.init(name: "batch_size", value: "4"), to: &config)
        #expect(config.learningRate == 0.0002)
        #expect(config.batchSize == 4)
    }

    @Test func rejectsUnsupportedChanges() {
        var config = TrainingConfig()
        #expect(throws: TrainingSettingsEditor.EditError.self) {
            try TrainingSettingsEditor.apply(.init(name: "api_key", value: "secret"), to: &config)
        }
    }
}
