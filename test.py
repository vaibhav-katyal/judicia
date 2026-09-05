from transformers import pipeline

pipe = pipeline(
    "fill-mask",
    model="law-ai/InLegalBERT"
)

# The accused was found [MASK] by the High Court.

# text = "The contract was declared [MASK]."

text = "The judge granted [MASK] to the accused."

result = pipe(text)

for r in result:
    print(f"{r['token_str']}  --> Score: {r['score']:.4f}")
    