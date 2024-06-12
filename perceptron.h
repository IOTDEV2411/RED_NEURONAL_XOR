#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <time.h>
#include <cmath>

class perceptron
{
public:
    perceptron(size_t number_in, double bias = 1);
    void set_weigths(std::vector<double> init_weigths);
    double core(std::vector<double> in_data);
    // double activation_function(double x);
    double sigmoide(double x);
    std::vector<double> weigths;
    double bias;
};

class perceptron_multicapa
{
public:
    perceptron_multicapa(std::vector<size_t> layers, double bias = 1);
    // Creamos nuestro "perceptron multicapa"
    // Tener presente que este tipo de red es
    // Una red de "perceptron simple"
    void set_weigths(std::vector<std::vector<std::vector<double>>> init_weights);
    std::vector<double> core(std::vector<double> in_data);
    void backpropagation(std::vector<double> x, std::vector<double> y);
    // Creamos nuestro vector de vectores de objetos tipo "perceptron"
    // en cada vector perceptron, se crearan el vector "weigths" (pesos)
    // y el bias respectivo por cada neurona de salida
    std::vector<std::vector<perceptron>> nets;
    // Definimos el atributo "layers", el cual es un vector
    // que nos indicara el numero de neuronas que se tendra en cada capa
    // Tener presente que la cantidad de neuronas en una capa
    // sera igual a la cantidad de salidas que tendra nuestra capa
    std::vector<size_t> layers;
    // Definimos el atributo "outputs", el cual sera un vector de vectores "double";
    // es decir, una matriz que contendra los valores de salida de cada capa
    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double>> d;
    double bias;
    double eta;
};