import joblib
model=joblib.load("trained.pk1")
y=input("Enter Years of Experience: ")
p=model.predict([[y]])
print("Predicted Salary: ",p)
