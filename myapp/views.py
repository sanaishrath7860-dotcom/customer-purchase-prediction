import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render, redirect
from django.contrib import messages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .models import UserRegistrationModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def index(request):
    return render(request, 'index.html')


def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request, "Registration successful! Please wait for account activation.")
        return render(request, 'UserRegistrations.html')
    return render(request, 'UserRegistrations.html')


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.error(request, 'Your account is not activated yet.')
                return render(request, 'UserLogin.html')
        except Exception:
            messages.error(request, 'Invalid Login ID or Password.')
            return render(request, 'UserLogin.html')
    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def AdminLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        if loginid == 'admin' and pswd == 'admin123':
            request.session['adminloggeduser'] = loginid
            users = UserRegistrationModel.objects.all()
            return render(request, 'admin/AdminHome.html', {'users': users})
        else:
            messages.error(request, 'Invalid Admin Credentials.')
            return render(request, 'AdminLogin.html')
    return render(request, 'AdminLogin.html')


def ActivateUser(request, uid):
    user = UserRegistrationModel.objects.get(id=uid)
    user.status = 'activated'
    user.save()
    users = UserRegistrationModel.objects.all()
    return render(request, 'admin/AdminHome.html', {'users': users})


def DeleteUser(request, uid):
    user = UserRegistrationModel.objects.get(id=uid)
    user.delete()
    users = UserRegistrationModel.objects.all()
    return render(request, 'admin/AdminHome.html', {'users': users})


def train_model(request):
    csv_path = os.path.join(BASE_DIR, 'ecommerce_prediction_dataset_with_target.csv')
    if not os.path.exists(csv_path):
        # Generate synthetic dataset if CSV not present
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'Coupon Discount Level': np.random.choice(['No Discount', '5%', '10%', '20%', '30%'], n),
            'Quantity Discount Level': np.random.choice(['No Discount', 'Buy 2 Get 1', 'Buy 3 Get 2'], n),
            'Time on Page (seconds)': np.random.randint(10, 600, n),
            'Added to Cart': np.random.choice(['Yes', 'No'], n),
            'Time of Interaction': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n),
            'Device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n),
            'Product Price ($)': np.round(np.random.uniform(5, 500, n), 2),
            'Product Rating (stars)': np.round(np.random.uniform(1, 5, n), 1),
        })
        df['Purchased'] = ((df['Added to Cart'] == 'Yes') &
                           (df['Product Rating (stars)'] >= 3.5)).astype(int)
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    df_ml = df.copy()
    categorical_columns = ['Coupon Discount Level', 'Quantity Discount Level',
                           'Added to Cart', 'Time of Interaction', 'Device']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        label_encoders[col] = le

    X = df_ml.drop(columns=[c for c in ['User ID', 'Product ID', 'Purchased'] if c in df_ml.columns])
    y = df_ml['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    static_dir = os.path.join(BASE_DIR, 'static')
    os.makedirs(static_dir, exist_ok=True)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend()
    plt.savefig(os.path.join(static_dir, 'roc_curve.png'))
    plt.close()

    joblib.dump(rf_model, os.path.join(BASE_DIR, 'random_forest_purchase_model.pkl'))
    joblib.dump(label_encoders, os.path.join(BASE_DIR, 'label_encoders.pkl'))

    context = {'accuracy': round(acc, 4), 'roc_auc': round(roc_auc, 4), 'report': report}
    return render(request, 'users/train.html', context)


def predict_view(request):
    prediction = None
    prob = None
    if request.method == 'POST':
        model_path = os.path.join(BASE_DIR, 'random_forest_purchase_model.pkl')
        enc_path = os.path.join(BASE_DIR, 'label_encoders.pkl')
        if not os.path.exists(model_path):
            messages.error(request, 'Model not trained yet. Please train the model first.')
            return render(request, 'users/predict.html', {})

        input_data = {
            'Coupon Discount Level': request.POST.get('coupon'),
            'Quantity Discount Level': request.POST.get('quantity'),
            'Time on Page (seconds)': int(request.POST.get('time', 0)),
            'Added to Cart': request.POST.get('cart'),
            'Time of Interaction': request.POST.get('interaction'),
            'Device': request.POST.get('device'),
            'Product Price ($)': float(request.POST.get('price', 0)),
            'Product Rating (stars)': float(request.POST.get('rating', 0)),
        }

        model = joblib.load(model_path)
        label_encoders = joblib.load(enc_path)

        for col in label_encoders:
            encoder = label_encoders[col]
            val = input_data[col]
            if val not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, val)
            input_data[col] = encoder.transform([val])[0]

        features = ['Coupon Discount Level', 'Quantity Discount Level',
                    'Time on Page (seconds)', 'Added to Cart', 'Time of Interaction',
                    'Device', 'Product Price ($)', 'Product Rating (stars)']
        X = pd.DataFrame([input_data])[features]
        prediction = model.predict(X)[0]
        prob = round(float(model.predict_proba(X)[0][1]) * 100, 2)

    result = None
    if prediction is not None:
        result = 'Customer Likely to Purchase ✅' if prediction == 1 else 'Customer Unlikely to Purchase ❌'

    return render(request, 'users/predict.html', {'prediction': result, 'probability': prob})
