from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django import forms

from .cls import TC

class SentenceForm(forms.Form):
    test_sentence = forms.CharField(label='Test sentence', max_length=100)

#tc_sentiment = TC(0)

def classifier(request):
    form = SentenceForm()
    # redirect from main page
    if request.method == 'REDIRECT':
        return HttpResponse("hah")
    '''
    global tc_sentiment
    score = tc_sentiment.classify("This food is so bad that i cant even smell")
    form = 0
    if request.method == 'POST':
        form = SentenceForm(request.POST)
        sample = form.data['test_sentence']
        score = tc_sentiment.classify(sample)
        if form.is_valid():
            return render(request, 'text_classifier/classifier.html', {'explanation':form.data,'form':form, 'score': score})
    else:
        form = SentenceForm()
    return render(request, 'text_classifier/classifier.html', {'explanation':'Enter your test sentence','form':form, 'score': 0})
    '''
    return render(request, 'text_classifier/page.html', {})

def part1(request):
    form = SentenceForm()
    return render(request, 'text_classifier/part1.html', {'form':form})

def part2(request):
    return HttpResponse("todo")