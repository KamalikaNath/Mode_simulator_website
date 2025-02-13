from django import forms
# from .models import UploadedData


class UploadFileForm(forms.Form):
    file = forms.FileField()

    class Meta:
        fields = ('file',)


# class UploadFileForm(forms.ModelForm):
#     class Meta:
#         model = UploadedData
#         fields = ('file',)
