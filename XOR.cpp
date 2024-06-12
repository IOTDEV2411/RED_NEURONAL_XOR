#include "perceptron.h"
#include <iostream>

int main()
{
    srand(time(NULL));
    rand();
    std::vector<size_t> neuron_layer;
    neuron_layer.push_back(2);
    neuron_layer.push_back(2);
    neuron_layer.push_back(1);
    perceptron_multicapa p(neuron_layer);

    std::vector<std::vector<std::vector<double>>> init_w;
    std::vector<std::vector<double>> hidden_layer;
    std::vector<std::vector<double>> output_layer;
    std::vector<double> neuron_01;
    neuron_01.push_back(-10);
    neuron_01.push_back(-10);
    neuron_01.push_back(15);
    hidden_layer.push_back(neuron_01);
    std::vector<double> neuron_02;
    neuron_02.push_back(15);
    neuron_02.push_back(15);
    neuron_02.push_back(-10);
    hidden_layer.push_back(neuron_02);
    std::vector<double> neuron_03;
    neuron_03.push_back(10);
    neuron_03.push_back(10);
    neuron_03.push_back(-15);
    output_layer.push_back(neuron_03);
    init_w.push_back(hidden_layer);
    init_w.push_back(output_layer);

    p.set_weigths(init_w);
    std::vector<double> combination_01;
    combination_01.push_back(0);
    combination_01.push_back(0);
    std::cout << "La salida de la compuerta XOR simulada con entrada 0,0 es: " << p.core(combination_01)[0] << std::endl;
    std::vector<double> combination_02;
    combination_02.push_back(0);
    combination_02.push_back(1);
    std::cout << "La salida de la compuerta XOR simulada con entrada 0,1 es: " << p.core(combination_02)[0] << std::endl;
    std::vector<double> combination_03;
    combination_03.push_back(1);
    combination_03.push_back(0);
    std::cout << "La salida de la compuerta XOR simulada con entrada 1,0 es: " << p.core(combination_03)[0] << std::endl;
    std::vector<double> combination_04;
    combination_04.push_back(1);
    combination_04.push_back(1);
    std::cout << "La salida de la compuerta XOR simulada con entrada 1,1 es: " << p.core(combination_04)[0] << std::endl;

    return 0;
};