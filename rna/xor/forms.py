from django import forms
from rna.xor.models import Xor


class XorForm(forms.Form):

    num_camadas = forms.IntegerField()
    bias = forms.BooleanField(initial=False, required=False)
    learningrate = forms.FloatField()
    momentum = forms.FloatField()
    epochs = forms.IntegerField()
    erro_max = forms.FloatField()
    peso_start = forms.FloatField()
    peso_end = forms.FloatField()
