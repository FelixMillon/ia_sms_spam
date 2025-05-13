# test_data.py

raw_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now.",
    "Hey, are we still on for dinner tonight?",
    "URGENT! Your account has been compromised. Send your details to secure@example.com.",
    "Don't forget to submit the report by 5PM.",
    "Free entry in 2 a weekly competition to win FA Cup final tickets! Text WIN to 80086",
    "Can you call me back when you're free?",
    "You have been selected for a chance to get a free iPhone! Click here.",
    "Hi mom, just checking in. Hope you're well!",
]

# 1 = spam, 0 = ham (vérifié manuellement)
true_labels = [1, 0, 1, 0, 1, 0, 1, 0]
