import json

from django.shortcuts import render, render_to_response
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from rna.xor.forms import XorForm


def index(request):
    context = {}
    if request.method == 'POST':
        form = XorForm(request.POST)
        if form.is_valid():
            return result(request, form)
    else:
        form = XorForm()
    context['form'] = form
    return render(request, 'inicio.html', context)


def result(request, form):
    dataset = SupervisedDataSet(2, 1)
    dados = form.cleaned_data

    # Adiciona a tabela XOR
    dataset.addSample([0, 0], [0])
    dataset.addSample([0, 1], [1])
    dataset.addSample([1, 0], [1])
    dataset.addSample([1, 1], [0])

    if dados['bias'] is None:
        bias = False
    else:
        bias = True

    # dimensões de entrada e saida, argumento 2 é a quantidade de camadas intermediárias
    network = buildNetwork(dataset.indim, int(dados['num_camadas']), dataset.outdim, bias=bias)
    trainer = BackpropTrainer(network, dataset, learningrate=float(dados['learningrate']), momentum=float(dados['momentum']))

    error = 1.00000000

    epocasPercorridas = 0

    errors = []
    it = []
    while epocasPercorridas < dados['epochs'] and error > dados['erro_max']:
        error = trainer.train()
        epocasPercorridas += 1
        errors.append(error)
        it.append(epocasPercorridas)
    graph = []
    idx = 0
    for e in errors:
        temp = []
        temp.append(idx)
        temp.append(e)
        idx +=1
        graph.append(temp)

    context = {'form': form.cleaned_data,
               'error': error,
               'graph': json.dumps(graph),
               'epocas': epocasPercorridas,
               'pesos_finais': network.params,
               'result00': network.activate([0, 0])[0],
               'result01': network.activate([0, 1])[0],
               'result10': network.activate([1, 0])[0],
               'result11': network.activate([1, 1])[0]}

    return render(request, 'result.html', context)
