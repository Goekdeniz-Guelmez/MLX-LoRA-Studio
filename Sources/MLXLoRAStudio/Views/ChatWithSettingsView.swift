import SwiftUI

struct ChatWithSettingsView: View {
    @Bindable var store: AppStore
    @State private var draft = ""
    @State private var keyDraft = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Chat with your settings").font(.title2.bold())
                    Text("Ask for training advice or tell the assistant to update your configuration.").foregroundStyle(.secondary)
                }
                Spacer()
                TextField("OpenAI model", text: $store.settingsChatModel).textFieldStyle(.roundedBorder).frame(width: 170)
            }.padding()
            Divider()
            if store.syntheticProviderKeyIsSet[.openai] != true {
                HStack {
                    SecureField("OpenAI API key", text: $keyDraft).textFieldStyle(.roundedBorder)
                    Button("Save to Keychain") { store.setSyntheticProviderKey(keyDraft, for: .openai); keyDraft = "" }
                        .disabled(keyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }.padding().background(.yellow.opacity(0.08))
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if store.settingsChatMessages.isEmpty {
                            ContentUnavailableView("Your training copilot", systemImage: "wand.and.stars", description: Text("Try “Review my current settings” or “Set a safer learning rate and explain why.”"))
                                .padding(.top, 80)
                        }
                        ForEach(store.settingsChatMessages) { message in
                            HStack {
                                if message.role == .user { Spacer(minLength: 90) }
                                Text(message.text).textSelection(.enabled).padding(12)
                                    .background(message.role == .user ? Color.accentColor.opacity(0.16) : Color.secondary.opacity(0.10), in: RoundedRectangle(cornerRadius: 14))
                                if message.role == .assistant { Spacer(minLength: 90) }
                            }.id(message.id)
                        }
                        if store.isSettingsChatResponding { HStack { ProgressView(); Text("Reviewing your settings…").foregroundStyle(.secondary); Spacer() } }
                    }.padding()
                }.onChange(of: store.settingsChatMessages.count) { _, _ in
                    if let id = store.settingsChatMessages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                }
            }
            if !store.settingsChatError.isEmpty { Text(store.settingsChatError).foregroundStyle(.red).font(.callout).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal) }
            HStack(alignment: .bottom) {
                TextField("Ask about or change your training settings…", text: $draft, axis: .vertical).lineLimit(1...5).textFieldStyle(.roundedBorder).onSubmit(send)
                Button(action: send) { Image(systemName: "arrow.up.circle.fill").font(.title2) }.buttonStyle(.plain)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || store.isSettingsChatResponding)
            }.padding()
        }.navigationTitle("Chat with Settings")
    }

    private func send() { let message = draft; draft = ""; Task { await store.sendSettingsChat(message) } }
}
