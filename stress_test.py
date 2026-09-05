from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/judicia-domain-model"

# =========================
# Load model
# =========================

print("Loading Judicia model...\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded!\n")


# =========================
# 50 HARD TEST CASES
# =========================

test_cases = [

    # =========================
    # CYBER - 10
    # =========================

    ("Someone hacked my Instagram account and changed my password.", "Cyber"),
    ("Someone took control of my WhatsApp account without my permission.", "Cyber"),
    ("My email account was hacked and the recovery details were changed.", "Cyber"),
    ("Someone created a fake Instagram profile using my photos.", "Cyber"),
    ("I received a phishing link and my online banking credentials were stolen.", "Cyber"),
    ("Someone stole my OTP and accessed my online account.", "Cyber"),
    ("My social media account was compromised and private messages were accessed.", "Cyber"),
    ("Someone is impersonating me online using my photographs.", "Cyber"),
    ("A person hacked my computer and encrypted all my files.", "Cyber"),
    ("Someone gained unauthorized access to my Facebook account.", "Cyber"),


    # =========================
    # CRIMINAL - 7
    # =========================

    ("Someone is sending me death threats on Instagram.", "Criminal"),
    ("My neighbour threatened to kill me during an argument.", "Criminal"),
    ("A person physically attacked me outside my house.", "Criminal"),
    ("I was assaulted by someone after a disagreement.", "Criminal"),
    ("Someone broke into my house and stole my belongings.", "Criminal"),
    ("A person threatened me with a knife.", "Criminal"),
    ("Someone is repeatedly threatening to harm my family.", "Criminal"),


    # =========================
    # PROPERTY - 9
    # =========================

    ("My brother is claiming ownership of our ancestral land.", "Property"),
    ("My brother refuses to give me my share of our ancestral property.", "Property"),
    ("My neighbour has illegally occupied part of my land.", "Property"),
    ("My tenant has stopped paying rent and refuses to vacate the property.", "Property"),
    ("My landlord is refusing to return my rental security deposit.", "Property"),
    ("Someone is trying to sell my property without my permission.", "Property"),
    ("There is a dispute over the boundary of my land with my neighbour.", "Property"),
    ("My co-owner is refusing to partition our jointly owned property.", "Property"),
    ("I discovered that another person is claiming ownership of my house.", "Property"),


    # =========================
    # FAMILY - 6
    # =========================

    ("My wife wants a divorce but I do not know what to do.", "Family"),
    ("My husband and I are separated and I want maintenance.", "Family"),
    ("My ex-wife is refusing to let me meet my child.", "Family"),
    ("I want legal custody of my child after separation.", "Family"),
    ("My parents want to legally adopt a child.", "Family"),
    ("My spouse and I are having a dispute about our child's custody.", "Family"),


    # =========================
    # LABOUR - 6
    # =========================

    ("My employer has not paid my salary for three months.", "Labour"),
    ("My company terminated me without giving me proper notice.", "Labour"),
    ("My employer is forcing me to work overtime without paying me.", "Labour"),
    ("My company deducted money from my salary without explaining why.", "Labour"),
    ("My employer refuses to give me my relieving letter.", "Labour"),
    ("I was denied maternity benefits by my employer.", "Labour"),


    # =========================
    # CIVIL - 4
    # =========================

    ("Someone breached a contract I signed with them.", "Civil"),
    ("A person owes me money under a written agreement and refuses to pay.", "Civil"),
    ("I paid someone for a service but they breached our agreement.", "Civil"),
    ("I want to recover money that someone owes me under a contract.", "Civil"),


    # =========================
    # CONSUMER - 4
    # =========================

    ("An online store delivered a completely different product than the one I ordered.", "Consumer"),
    ("A company refuses to repair my phone even though it is under warranty.", "Consumer"),
    ("My internet provider keeps charging me for a service I cancelled.", "Consumer"),
    ("An airline cancelled my flight and is refusing to provide the promised refund.", "Consumer"),


    # =========================
    # CONSTITUTIONAL - 4
    # =========================

    ("The government is restricting my freedom of speech without giving any reason.", "Constitutional"),
    ("A government authority is treating me differently because of my religion.", "Constitutional"),
    ("The government is unlawfully restricting my right to peacefully protest.", "Constitutional"),
    ("A government authority collected my personal information without legal justification.", "Constitutional"),
]


# =========================
# Prediction helper
# =========================

def get_label(prediction):

    # Transformers versions can store keys as strings or integers
    if prediction in model.config.id2label:
        return model.config.id2label[prediction]

    return model.config.id2label[str(prediction)]


# =========================
# Run test
# =========================

true_labels = []
predicted_labels = []

correct = 0

print("=" * 80)
print("JUDICIA REAL-WORLD STRESS TEST")
print("=" * 80)

for i, (text, expected) in enumerate(test_cases, start=1):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    confidence = probabilities[0][prediction].item()

    predicted = get_label(prediction)

    true_labels.append(expected)
    predicted_labels.append(predicted)

    if predicted == expected:
        correct += 1
        status = "✓"
    else:
        status = "✗"

    print(
        f"{i:02d}. {status} | "
        f"Expected: {expected:<15} | "
        f"Predicted: {predicted:<15} | "
        f"Confidence: {confidence * 100:6.2f}%"
    )

    print(f"    {text}")


# =========================
# Overall accuracy
# =========================

accuracy = correct / len(test_cases)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(f"\nCorrect: {correct}/{len(test_cases)}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# =========================
# Classification report
# =========================

labels = [
    "Civil",
    "Constitutional",
    "Consumer",
    "Criminal",
    "Cyber",
    "Family",
    "Labour",
    "Property"
]

print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=labels,
        zero_division=0,
        digits=3
    )
)


# =========================
# Confusion Matrix
# =========================

print("Confusion Matrix:")

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=labels
)

print("\nLabels order:")
print(labels)

print("\n")
print(cm)