from predict_stress import predict_stress

test_sentences = [
    "My brain feels like it’s running in circles all day.",
    "Everything seems fine, I’m chill about the exams.",
    "I can’t even bring myself to open my books anymore.",
    "The constant dea"
    "dlines are eating away my peace.",
    "I’m okay, just a bit jittery before the presentation.",
    "I’ve been crying almost every night lately.",
    "I’m doing fine, just busy juggling tasks.",
    "Sometimes I forget to eat because of all the work.",
    "My mind feels foggy and I can’t think clearly.",
    "I’m proud of how I’m handling everything this mo"
    
]

for text in test_sentences:
    result = predict_stress(text)
    print(f"\nInput: {text}")
    print("→", result)



