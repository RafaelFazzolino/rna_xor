from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Xor(models.Model):

    num_camadas = models.IntegerField(validators=[MinValueValidator(1),
                                       MaxValueValidator(6)])
    bias = models.BooleanField(default=False)
    learningrate = models.FloatField(validators=[MinValueValidator(0),
                                       MaxValueValidator(1)])
    momentum = models.FloatField(validators=[MinValueValidator(0),
                                       MaxValueValidator(1)])
    epochs = models.IntegerField()
