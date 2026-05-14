from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.UserRegisterActions, name='register'),
    path('login/', views.UserLoginCheck, name='login'),
    path('home/', views.UserHome, name='home'),
    path('adminlogin/', views.AdminLoginCheck, name='adminlogin'),
    path('activate/<int:uid>/', views.ActivateUser, name='activate'),
    path('delete/<int:uid>/', views.DeleteUser, name='delete'),
    path('train/', views.train_model, name='train'),
    path('predict/', views.predict_view, name='predict'),
]
