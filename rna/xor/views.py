from django.shortcuts import render
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from rna.xor.forms import XorForm
from rna.xor.my_backprop import myOwn_BackpropTrainer


def index(request):
    context = {}
    if request.method == 'POST':
        form = XorForm(request.POST)
        if form.is_valid():
            return result(request, form)
    else:
        form = XorForm()
    context['form'] = form
    return render(request, 'index.html', context)


def result(request, form):
    dataset = SupervisedDataSet(2, 1)
    dados = form.cleaned_data

    # Adiciona a tabela XOR
    dataset.addSample([0, 0], [0])
    dataset.addSample([0, 1], [1])
    dataset.addSample([1, 0], [1])
    dataset.addSample([1, 1], [0])

    # dimensões de entrada e saida, argumento 2 é a quantidade de camadas intermediárias
    network = buildNetwork(dataset.indim, dados['num_camadas'], dataset.outdim, dados['bias'])
    trainer = BackpropTrainer(network, dataset, dados['learningrate'], dados['momentum'])
    trainer.trainEpochs(dados['epochs'])

    max_error = 1
    error = 0.00001
    # epocas = 1000

    # inicializando contador de epocas
    epocasPercorridas = 0

    errors = []
    it = []
    while epocasPercorridas < dados['epochs']:
        error = trainer.train()
        epocasPercorridas += 1
        errors.append(error)
        it.append(epocasPercorridas)
        if error == 0:
            break

        print("\n\nPesos finais: ", network.params)
        print("\nErro final: ", error)

        print("\n\nTotal de epocas percorridas: ", epocasPercorridas)

        print('\n\n1 XOR 1: Esperado = 0, Calculado = ', network.activate([1, 1])[0])
        print('1 XOR 0: Esperado = 1, Calculado =', network.activate([1, 0])[0])
        print('0 XOR 1: Esperado = 1, Calculado =', network.activate([0, 1])[0])
        print('0 XOR 0: Esperado = 0, Calculado =', network.activate([0, 0])[0])
        print('Lista de erros', len(errors))
        print('Lista de it', len(it))

    context = {'form':form.cleaned_data,
               'error': error,
               'errors': errors,
               'iteracoes': it,
               'epocas': epocasPercorridas,
               'pesos_finais': network.params,
               'result00': network.activate([0, 0])[0],
               'result01': network.activate([0, 1])[0],
               'result10': network.activate([1, 0])[0],
               'result11': network.activate([1, 1])[0]}

    return render(request, 'result.html', context)