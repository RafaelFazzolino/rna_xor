from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Xor(models.Model):
    name = models.CharField(max_length=30)
    descricao = models.TextField(null=True, blank=True)
    num_camadas = models.IntegerField(validators=[MinValueValidator(1),
                                       MaxValueValidator(6)])
    bias = models.BooleanField()
    learningrate = models.FloatField(validators=[MinValueValidator(0),
                                       MaxValueValidator(1)])
    momentum = models.FloatField(validators=[MinValueValidator(0),
                                       MaxValueValidator(1)])
    epochs = models.IntegerField()

    def __str__(self):
        return self.name