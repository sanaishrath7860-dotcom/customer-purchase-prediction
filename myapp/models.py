from django.db import models

class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(max_length=15)
    email = models.EmailField()
    locality = models.CharField(max_length=100)
    address = models.TextField()
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=20, default='waiting')

    def __str__(self):
        return self.name
