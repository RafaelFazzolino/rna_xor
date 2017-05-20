from django import forms
from rna.xor.models import Xor


class XorForm(forms.ModelForm):
    class Meta:
        model = Xor
        fields = ('name', 'descricao', 'num_camadas', 'bias', 'learningrate', 'momentum', 'epochs',)