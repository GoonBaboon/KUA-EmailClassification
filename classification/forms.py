from django import forms

class EmailForm(forms.Form):
    email_text = forms.CharField(label='Enter Email Text', widget=forms.Textarea)
