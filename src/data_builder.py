"""
data_builder.py
Dataset creation, cleaning, encoding, and train/test split.
"""
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ANSWERS = {
    "working_hours": (
        "Core working hours are 9am to 6pm, Monday to Friday. "
        "Saturday and Sunday are off. Lunch break is from 1pm to 2pm. "
        "Flexible timing can be discussed with your supervisor after the first month."
    ),
    "leave_request": (
        "Submit a leave request through the HR portal at least 3 days in advance. "
        "Your supervisor will approve it via email. "
        "For sick leave, inform your supervisor by 9am on the same day. "
        "Emergency leave: contact HR directly at hr@company.com."
    ),
    "stipend_query": (
        "Intern stipends are processed on the 25th of every month "
        "and credited within 2 working days. "
        "Ensure your bank details are submitted to HR in your first week. "
        "You will receive a digital pay slip via email."
    ),
    "it_support": (
        "Visit the IT helpdesk on Floor 2 with your employee ID. "
        "For urgent issues call ext 200 or email it-support@company.com. "
        "Laptop setup takes approximately 2 hours on your first day. "
        "VPN access is configured by IT during onboarding."
    ),
    "credential_issue": (
        "Your login credentials are sent to your personal email before joining. "
        "If not received, contact IT at it-support@company.com. "
        "Password resets can be done via the portal login page. "
        "Account lockouts are resolved within 2 hours by IT support."
    ),
    "hr_policy": (
        "Interns follow business casual dress code. "
        "You must sign an NDA on your first day. "
        "Internship certificates are issued within 2 weeks of completion. "
        "Performance reviews happen at the end of internship."
    ),
    "general_query": (
        "Your assigned buddy is your first point of contact. "
        "HR is reachable at hr@company.com or Floor 3. "
        "Your access card is issued on day one at reception. "
        "Emergency contact: front desk at ext 100."
    ),
}

FAQ_DATA = [
    # working_hours
    {"user_input": "What are the office working hours?",           "intent": "working_hours"},
    {"user_input": "What time does the office open?",              "intent": "working_hours"},
    {"user_input": "When does the workday end?",                   "intent": "working_hours"},
    {"user_input": "What are the office timings?",                 "intent": "working_hours"},
    {"user_input": "Is there flexible timing for interns?",        "intent": "working_hours"},
    {"user_input": "Can I work from home?",                        "intent": "working_hours"},
    {"user_input": "What days do we work?",                        "intent": "working_hours"},
    {"user_input": "Do we work on Saturdays?",                     "intent": "working_hours"},
    {"user_input": "Is Sunday a holiday?",                         "intent": "working_hours"},
    {"user_input": "What is the lunch break timing?",              "intent": "working_hours"},
    {"user_input": "How many hours do interns work daily?",        "intent": "working_hours"},
    {"user_input": "Can I leave early sometimes?",                 "intent": "working_hours"},
    {"user_input": "Are there any half days?",                     "intent": "working_hours"},
    {"user_input": "What time should I arrive at the office?",     "intent": "working_hours"},
    # leave_request
    {"user_input": "How do I apply for leave?",                    "intent": "leave_request"},
    {"user_input": "How many leaves do I get as an intern?",       "intent": "leave_request"},
    {"user_input": "Can I take a day off?",                        "intent": "leave_request"},
    {"user_input": "Who approves my leave?",                       "intent": "leave_request"},
    {"user_input": "What is the leave policy for interns?",        "intent": "leave_request"},
    {"user_input": "How much notice do I need to give for leave?", "intent": "leave_request"},
    {"user_input": "Can I take emergency leave?",                  "intent": "leave_request"},
    {"user_input": "I am sick and cannot come to office",          "intent": "leave_request"},
    {"user_input": "How do I inform about my absence?",            "intent": "leave_request"},
    {"user_input": "Is medical leave paid?",                       "intent": "leave_request"},
    {"user_input": "Can I take leave in my first week?",           "intent": "leave_request"},
    {"user_input": "How do I fill the leave application form?",    "intent": "leave_request"},
    {"user_input": "What counts as an excused absence?",           "intent": "leave_request"},
    {"user_input": "Can I take half day leave?",                   "intent": "leave_request"},
    # stipend_query
    {"user_input": "When will I receive my stipend?",              "intent": "stipend_query"},
    {"user_input": "How much is the intern stipend?",              "intent": "stipend_query"},
    {"user_input": "When is payday?",                              "intent": "stipend_query"},
    {"user_input": "I have not received my salary yet",            "intent": "stipend_query"},
    {"user_input": "How do I submit my bank details?",             "intent": "stipend_query"},
    {"user_input": "When does salary get credited?",               "intent": "stipend_query"},
    {"user_input": "What is the payment date for interns?",        "intent": "stipend_query"},
    {"user_input": "Is the stipend paid weekly or monthly?",       "intent": "stipend_query"},
    {"user_input": "Do interns get paid for overtime?",            "intent": "stipend_query"},
    {"user_input": "Will I get a pay slip?",                       "intent": "stipend_query"},
    {"user_input": "Is the stipend taxable?",                      "intent": "stipend_query"},
    {"user_input": "How do I check if my salary is credited?",     "intent": "stipend_query"},
    {"user_input": "My stipend amount is incorrect",               "intent": "stipend_query"},
    {"user_input": "Do I get travel allowance?",                   "intent": "stipend_query"},
    # it_support
    {"user_input": "My laptop is not working",                     "intent": "it_support"},
    {"user_input": "How do I get my laptop set up?",               "intent": "it_support"},
    {"user_input": "Who do I contact for IT issues?",              "intent": "it_support"},
    {"user_input": "I cannot connect to the office wifi",          "intent": "it_support"},
    {"user_input": "Where is the IT helpdesk?",                    "intent": "it_support"},
    {"user_input": "My computer is not turning on",                "intent": "it_support"},
    {"user_input": "I need a software installed on my laptop",     "intent": "it_support"},
    {"user_input": "The office printer is not working",            "intent": "it_support"},
    {"user_input": "How do I connect to the VPN?",                 "intent": "it_support"},
    {"user_input": "I need access to the company drive",           "intent": "it_support"},
    {"user_input": "My screen is broken",                          "intent": "it_support"},
    {"user_input": "Which software do I need for my work?",        "intent": "it_support"},
    {"user_input": "I cannot access my work emails",               "intent": "it_support"},
    {"user_input": "How do I join the office Slack?",              "intent": "it_support"},
    # credential_issue
    {"user_input": "I forgot my password",                         "intent": "credential_issue"},
    {"user_input": "I have not received my login credentials",     "intent": "credential_issue"},
    {"user_input": "How do I reset my password?",                  "intent": "credential_issue"},
    {"user_input": "I cannot log into my work account",            "intent": "credential_issue"},
    {"user_input": "My email account is not working",              "intent": "credential_issue"},
    {"user_input": "I did not get my office email yet",            "intent": "credential_issue"},
    {"user_input": "How do I access the company portal?",          "intent": "credential_issue"},
    {"user_input": "My account is locked",                         "intent": "credential_issue"},
    {"user_input": "I am getting an invalid password error",       "intent": "credential_issue"},
    {"user_input": "How long does it take to get credentials?",    "intent": "credential_issue"},
    {"user_input": "I never received my welcome email",            "intent": "credential_issue"},
    {"user_input": "Can I change my work email password?",         "intent": "credential_issue"},
    {"user_input": "I cannot log into the HR portal",             "intent": "credential_issue"},
    {"user_input": "My two factor authentication is not working",  "intent": "credential_issue"},
    # hr_policy
    {"user_input": "What is the dress code?",                      "intent": "hr_policy"},
    {"user_input": "What should I wear to the office?",            "intent": "hr_policy"},
    {"user_input": "What is the intern code of conduct?",          "intent": "hr_policy"},
    {"user_input": "How do I get my internship certificate?",      "intent": "hr_policy"},
    {"user_input": "Will I get a letter of recommendation?",       "intent": "hr_policy"},
    {"user_input": "What happens at the end of internship?",       "intent": "hr_policy"},
    {"user_input": "Can my internship be extended?",               "intent": "hr_policy"},
    {"user_input": "Is there a probation period for interns?",     "intent": "hr_policy"},
    {"user_input": "Will I get a performance review?",             "intent": "hr_policy"},
    {"user_input": "What are the rules for interns?",              "intent": "hr_policy"},
    {"user_input": "Can my internship convert to a full time job?","intent": "hr_policy"},
    {"user_input": "What is the confidentiality policy?",          "intent": "hr_policy"},
    {"user_input": "Do I need to sign an NDA?",                    "intent": "hr_policy"},
    {"user_input": "What documents do I need to submit on joining?","intent": "hr_policy"},
    # general_query
    {"user_input": "Who is my supervisor?",                        "intent": "general_query"},
    {"user_input": "Who do I talk to if I have a problem?",        "intent": "general_query"},
    {"user_input": "How do I reach HR?",                           "intent": "general_query"},
    {"user_input": "Where is the HR office?",                      "intent": "general_query"},
    {"user_input": "Who is my point of contact?",                  "intent": "general_query"},
    {"user_input": "How do I get my ID card?",                     "intent": "general_query"},
    {"user_input": "Where do I park my vehicle?",                  "intent": "general_query"},
    {"user_input": "How do I get access to the office building?",  "intent": "general_query"},
    {"user_input": "Is there a cafeteria in the office?",          "intent": "general_query"},
    {"user_input": "What is the office address?",                  "intent": "general_query"},
    {"user_input": "Is there a gym or recreational area?",         "intent": "general_query"},
    {"user_input": "How do I get my access card?",                 "intent": "general_query"},
    {"user_input": "Who do I contact in case of an emergency?",    "intent": "general_query"},
    {"user_input": "Is there a buddy program for new interns?",    "intent": "general_query"},
]

TICKET_DATA = [
    {"user_input": "What time should I be in office?",                     "intent": "working_hours"},
    {"user_input": "Are we working this Saturday?",                        "intent": "working_hours"},
    {"user_input": "Do interns have flexible hours?",                      "intent": "working_hours"},
    {"user_input": "When does lunch break start?",                         "intent": "working_hours"},
    {"user_input": "Is Friday a half day?",                                "intent": "working_hours"},
    {"user_input": "I have a family emergency can I take leave tomorrow?",  "intent": "leave_request"},
    {"user_input": "How do I fill leave form online?",                     "intent": "leave_request"},
    {"user_input": "My leave was rejected what should I do?",              "intent": "leave_request"},
    {"user_input": "Can I take leave without prior notice?",               "intent": "leave_request"},
    {"user_input": "I need a sick day today",                              "intent": "leave_request"},
    {"user_input": "Its the 27th and I have not received my stipend",      "intent": "stipend_query"},
    {"user_input": "My bank account details changed how do I update them?", "intent": "stipend_query"},
    {"user_input": "Will I get paid for the extra hours I worked?",        "intent": "stipend_query"},
    {"user_input": "I did not receive my payslip this month",              "intent": "stipend_query"},
    {"user_input": "When exactly will salary hit my account?",             "intent": "stipend_query"},
    {"user_input": "My laptop screen is flickering",                       "intent": "it_support"},
    {"user_input": "I cannot install the required software on my machine", "intent": "it_support"},
    {"user_input": "Office wifi keeps disconnecting",                      "intent": "it_support"},
    {"user_input": "My keyboard is not working properly",                  "intent": "it_support"},
    {"user_input": "I need access to the shared company drive",            "intent": "it_support"},
    {"user_input": "I never received my login details before joining",     "intent": "credential_issue"},
    {"user_input": "My password expired on day one",                       "intent": "credential_issue"},
    {"user_input": "I am locked out of my account after 3 wrong attempts", "intent": "credential_issue"},
    {"user_input": "Two factor authentication code is not arriving",       "intent": "credential_issue"},
    {"user_input": "I cannot access the HR portal with my credentials",    "intent": "credential_issue"},
    {"user_input": "What documents should I bring on my first day?",       "intent": "hr_policy"},
    {"user_input": "Will I receive a certificate after internship ends?",  "intent": "hr_policy"},
    {"user_input": "Is there a chance of getting hired full time?",        "intent": "hr_policy"},
    {"user_input": "What is the NDA policy for interns?",                  "intent": "hr_policy"},
    {"user_input": "Can I share company information on LinkedIn?",         "intent": "hr_policy"},
    {"user_input": "Where do I sit on my first day?",                      "intent": "general_query"},
    {"user_input": "How do I get my access card for the building?",        "intent": "general_query"},
    {"user_input": "Is there a cafeteria in the office building?",         "intent": "general_query"},
    {"user_input": "Who do I report to if my supervisor is absent?",       "intent": "general_query"},
    {"user_input": "How do I get my intern ID card made?",                 "intent": "general_query"},
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dataset():
    data = FAQ_DATA + TICKET_DATA
    df = pd.DataFrame(data)
    df["answer"]        = df["intent"].map(ANSWERS)
    df["cleaned_input"] = df["user_input"].apply(clean_text)
    le = LabelEncoder()
    df["intent_label"]  = le.fit_transform(df["intent"])
    return df, le


def split_dataset(df, test_size=0.2):
    X = df["cleaned_input"]
    y = df["intent_label"]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


if __name__ == "__main__":
    df, le = build_dataset()
    print(f"Dataset shape : {df.shape}")
    print(f"Intents       : {list(le.classes_)}")
    print(df["intent"].value_counts())
    df.to_csv("../data/intern_dataset_full.csv", index=False)
    print("Saved to data/intern_dataset_full.csv")
