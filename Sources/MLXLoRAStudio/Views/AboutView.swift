import SwiftUI

struct AboutView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HeaderView(
                    title: "About",
                    subtitle: "Credits, citations, and the reason this app exists.",
                    symbol: "info.circle"
                )

                VStack(alignment: .leading, spacing: 16) {
                    Text("MLX LoRA Studio is a native desktop app for fine-tuning language models on Apple-silicon with LoRA, DoRA, supervised fine-tuning, preference optimization, reinforcement-style training loops, synthetic data generation, and live run monitoring.")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("I built this to make the MLX fine-tuning workflow feel less like a stack of scripts and more like a focused studio: choose a model, pick data, tune the important knobs, launch training, and watch the run without losing the command-line transparency underneath.")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .formBlock()

                VStack(alignment: .leading, spacing: 12) {
                    SectionTitle("How To Cite This Work")
                    
                    Text("If you use this app or its training workflow in a project, please cite the following:")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.bottom, 8)
                    
                    BibTeXEntry(
                        key: "mlx2023",
                        type: "software",
                        fields: [
                            ("author", "Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert"),
                            ("title", "{MLX}: Efficient and flexible machine learning on Apple silicon"),
                            ("url", "https://github.com/ml-explore"),
                            ("version", "0.0"),
                            ("year", "2023")
                        ]
                    )
                    
                    BibTeXEntry(
                        key: "gulmez2025mlxlmlora",
                        type: "software",
                        fields: [
                            ("author", "Gökdeniz Gülmez"),
                            ("title", "{MLX-LM-LoRA}: Train LLMs on Apple silicon with MLX and the Hugging Face Hub"),
                            ("url", "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora"),
                            ("version", "0.1.0"),
                            ("year", "2025")
                        ]
                    )
                    
                    BibTeXEntry(
                        key: "gulmez2026mlxlorastudio",
                        type: "software",
                        fields: [
                            ("author", "Gökdeniz Gülmez"),
                            ("title", "{MLX-LoRA-Studio}: A native Mac App for LLM fine-tuning on Apple Silicon — fully on-device, fully open source."),
                            ("url", "https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio.git"),
                            ("version", "1.0.0"),
                            ("year", "2026")
                        ]
                    )
                }
                .formBlock()

                VStack(alignment: .leading, spacing: 12) {
                    SectionTitle("Support This Project and Me")
                    
                    Link(destination: URL(string: "https://github.com/sponsors/Goekdeniz-Guelmez")!) {
                        HStack(spacing: 12) {
                            Image(systemName: "heart.fill")
                                .font(.title3)
                                .foregroundStyle(.pink)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Sponsor on GitHub")
                                    .font(.headline)
                                    .foregroundStyle(.primary)
                                Text("Support the development of MLX LoRA Studio")
                                    .font(.callout)
                                    .foregroundStyle(.secondary)
                            }
                            
                            Spacer()
                            
                            Image(systemName: "arrow.up.forward")
                                .font(.callout)
                                .foregroundStyle(.tertiary)
                        }
                        .padding(16)
                        .background(
                            LinearGradient(
                                colors: [Color.pink.opacity(0.15), Color.purple.opacity(0.15)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            in: RoundedRectangle(cornerRadius: 12, style: .continuous)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(Color.pink.opacity(0.3), lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
                .formBlock()
            }
            .padding(24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .liquidGlass(cornerRadius: 18)
        .padding(16)
        .navigationTitle("About")
    }
}

private struct BibTeXEntry: View {
    let key: String
    let type: String
    let fields: [(String, String)]
    
    @State private var showCopied = false
    
    private var bibtexString: String {
        var result = "@\(type){\(key),\n"
        for field in fields {
            result += "  \(field.0) = {\(field.1)},\n"
        }
        result += "}"
        return result
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "doc.text.fill")
                    .font(.callout)
                    .foregroundStyle(.tint)
                
                Text("@\(type){\(key),")
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.semibold)
                    .foregroundStyle(.primary)
                
                Spacer()
                
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(bibtexString, forType: .string)
                    showCopied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        showCopied = false
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: showCopied ? "checkmark" : "doc.on.doc")
                            .font(.caption)
                        Text(showCopied ? "Copied" : "Copy")
                            .font(.caption)
                    }
                    .foregroundStyle(showCopied ? .green : .secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                .buttonStyle(.plain)
            }
            
            VStack(alignment: .leading, spacing: 2) {
                ForEach(fields, id: \.0) { field in
                    HStack(alignment: .top, spacing: 0) {
                        Text("  \(field.0)")
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(.secondary)
                        
                        Text(" = ")
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(.tertiary)
                        
                        Text("{\(field.1)}")
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(.primary)
                        
                        Text(",")
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            
            Text("}")
                .font(.system(.body, design: .monospaced))
                .fontWeight(.semibold)
                .foregroundStyle(.primary)
        }
        .textSelection(.enabled)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
