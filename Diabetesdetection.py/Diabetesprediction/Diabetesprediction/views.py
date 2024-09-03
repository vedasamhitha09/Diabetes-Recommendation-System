from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    if request.method == 'POST':
        data = pd.read_csv(r"C:\Users\V V Samhitha\OneDrive\Desktop\ML project\diabetes.csv")
        x = data.drop("Outcome", axis=1)
        y = data['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)

        # Get the input data from the form
        val1 = float(request.POST.get('n1', 0))
        val2 = float(request.POST.get('n2', 0))
        val3 = float(request.POST.get('n3', 0))
        val4 = float(request.POST.get('n4', 0))
        val5 = float(request.POST.get('n5', 0))
        val6 = float(request.POST.get('n6', 0))
        val7 = float(request.POST.get('n7', 0))
        val8 = float(request.POST.get('n8', 0))

        # Predict the outcome
        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
        result1 = "positive" if pred[0] == 1 else "negative"

        # Return the result along with the form data
        return render(request, 'predict.html', {
            "result2": result1,
            "n1": val1,
            "n2": val2,
            "n3": val3,
            "n4": val4,
            "n5": val5,
            "n6": val6,
            "n7": val7,
            "n8": val8
        })
    else:
        return render(request, 'predict.html')
