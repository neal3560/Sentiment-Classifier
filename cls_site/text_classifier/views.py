from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django import forms

from .cls import TC

class SentenceForm(forms.Form):
    input = 'Input Sentence'
    test_sentence = forms.CharField(label='', max_length=100,widget=forms.TextInput(attrs={'placeholder': input,}))

tc_sentiment = TC(0)
tc_toxic = TC(1)

def classifier(request):
    form = SentenceForm()
    # redirect from main page
    if request.method == 'REDIRECT':
        return HttpResponse("hah")
    return render(request, 'text_classifier/page.html', {})

def part1(request):
    global tc_sentiment
    form = SentenceForm()
    if request.method == 'POST':
        form = SentenceForm(request.POST)
        sample = form.data['test_sentence']
        info = tc_sentiment.classify(sample)
        if form.is_valid():
            return render(request, 'text_classifier/part1.html', {'sentence':form.data,'form':form, 'info': info, 'action':'result'})
    return render(request, 'text_classifier/part1.html', {'form':form, 'action':'input'})

def part2(request):
    global tc_toxic
    form = SentenceForm()
    if request.method == 'POST':
        form = SentenceForm(request.POST)
        sample = form.data['test_sentence']
        info = tc_toxic.classify(sample)
        if form.is_valid():
            return render(request, 'text_classifier/part2.html', {'sentence':form.data,'form':form, 'info': info, 'action':'result'})
    return render(request, 'text_classifier/part2.html', {'form':form, 'action':'input'})
