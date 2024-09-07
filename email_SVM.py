import os
import glob
import email
from tkinter import *
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

def load_emails_from_folder(folder):
    emails = []
    for filepath in glob.glob(os.path.join(folder, '*')):
        with open(filepath, 'r', encoding='latin1') as f:
            try:
                msg = email.message_from_file(f)
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            emails.append(part.get_payload())
                else:
                    emails.append(msg.get_payload())
            except Exception as e:
                print(f"Error reading email: {e}")
    return emails

ham_folder = 'datasets/ham'
spam_folder = 'datasets/spam'

ham_emails = load_emails_from_folder(ham_folder)
spam_emails = load_emails_from_folder(spam_folder)

ham_emails_clean = [preprocess_text(email) for email in ham_emails if preprocess_text(email).strip() != '']
spam_emails_clean = [preprocess_text(email) for email in spam_emails if preprocess_text(email).strip() != '']

ham_labels = [0] * len(ham_emails_clean)
spam_labels = [1] * len(spam_emails_clean)

emails = ham_emails_clean + spam_emails_clean
labels = ham_labels + spam_labels

if not emails:
    raise ValueError("No valid documents to process")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Use SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_spam():
    email_text = email_input.get()
    message_text = message_input.get("1.0", END)
    
    preprocessed_message = preprocess_text(message_text)
    
    if preprocessed_message.strip() == '':
        messagebox.showwarning("Warning", "Input message is empty.")
        return
    
    message_tfidf = vectorizer.transform([preprocessed_message])
    prediction = classifier.predict(message_tfidf)[0]
    
    result = "This email is SPAM!" if prediction == 1 else "This email is NOT SPAM!"
    messagebox.showinfo("Prediction Result", result)

root = Tk()
root.title("Email Spam Detector")

email_label = Label(root, text="Email Address:")
email_label.grid(row=0, column=0, padx=10, pady=10)
email_input = Entry(root, width=50)
email_input.grid(row=0, column=1, padx=10, pady=10)

message_label = Label(root, text="Message:")
message_label.grid(row=1, column=0, padx=10, pady=10)
message_input = Text(root, height=10, width=50)
message_input.grid(row=1, column=1, padx=10, pady=10)

predict_button = Button(root, text="Predict", command=predict_spam)
predict_button.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()
