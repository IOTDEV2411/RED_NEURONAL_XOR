#include "perceptron.h"

double random_function()
{
    // Implementamos una funcion aleatoria para que
    // nos genere un numero aleatoria, que sera
    // usada para inicializar el vector "weigths" (pesos)
    return (2.0 * (double)rand() / RAND_MAX);
};

perceptron::perceptron(size_t number_in, double bias)
{

    // Constructor para nuestra clase "perceptron"
    // Con este constructor, creamos nuestro atributo "weigths"
    // asi como el atributo "bias" de la clase "perceptron"

    // Inicializamos los atributos de la clase "perceptron"
    // Comenzamos con el bias
    this->bias = bias;

    // Aqui, indicamos la cantidad de elementos que tendra nuestro vector "weigths" (pesos)
    this->weigths.resize(number_in + 1);

    // Proseguimos llenando cada uno de los elementos de nuestro vector "weigths" (pesos)
    // mediante el metodo "generate()", indicando el primer y el ultimo elemento de nuestro
    // vector, asi como la funcion que usaremos para "generar" cada elemento de nuestro vector.
    // Tener presente que la cantidad de elementos de nuestro vector "weigths" sera igual
    // a la cantidad de entradas de nuestra red perceptron simple.
    generate(this->weigths.begin(), this->weigths.end(), random_function);
};

void perceptron::set_weigths(std::vector<double> init_weigths)
{
    // implementamos la funcion para asignar los valores
    // del vector "weights"
    weigths = init_weigths;
}

double perceptron::sigmoide(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// double perceptron::weigthed_sum(vector<double> in_data){

// };

double perceptron::core(std::vector<double> in_data)
{
    // Funcion que se encargara de implementar las tareas
    // del nucleo (core) de nuestro perceptron

    // Aniadimos al vector de entrada el bias respectivo
    in_data.push_back(bias);
    // Realizamos la suma ponderada del vector de entrada con los pesos respectivos
    // Para eso, hacemos uso de la funcion "inner_product"
    double sum_of_weigths = inner_product(in_data.begin(), in_data.end(), weigths.begin(), double(0.0));
    // Aplicamos la funcion sigmoide como funcion de activacion
    return sigmoide(sum_of_weigths);
}

perceptron_multicapa::perceptron_multicapa(std::vector<size_t> layers, double bias)
{
    // Constructor para nuestra clase "perceptron_multicapa"
    this->layers = layers;
    // Definimos el atributo "bias"
    this->bias = bias;

    for (size_t i = 0; i < layers.size(); i++)
    {
        // Por cada iteracion, inicializaremos cada una de las filas
        // de nuestra matriz "outputs" con una longitud igual a la indicada
        // en su respectivo elemento en el vector "layers" y con valor igual a 0.0
        this->outputs.push_back(std::vector<double>(this->layers[i], 0.0));
        // Inicializamos nuestra red "perceptron multicapa"
        // con "perceptron simple" vacio
        this->nets.push_back(std::vector<perceptron>());

        if (i > 0)
        {
            for (size_t j = 0; j < layers[i]; j++)
            {
                this->nets[i].push_back(perceptron(this->layers[i - 1], this->bias));
            };
        };
    };
}

void perceptron_multicapa::set_weigths(std::vector<std::vector<std::vector<double>>> init_weights)
{

    for (size_t i = 0; i < init_weights.size(); i++)
    {
        for (size_t j = 0; j < init_weights[i].size(); j++)
        {
            this->nets[i + 1][j].set_weigths(init_weights[i][j]);
        };
    };
}

std::vector<double> perceptron_multicapa::core(std::vector<double> in_data)
{
    outputs[0] = in_data;

    for (size_t i = 1; i < nets.size(); i++)
    {
        for (size_t j = 0; j < layers[i]; j++)
        {
            this->outputs[i][j] = this->nets[i][j].core(this->outputs[i - 1]);
        };
    };

    // Devolvemos los valores de salida de la red, las cuales estan contenidas
    // en el ultimo elemento de nuestro vector de vectores "outputs"
    return outputs.back();
}

void perceptron_multicapa::backpropagation(std::vector<double> x, std::vector<double> y)
{

    // En este bloque de codigo, implementaremos los pasos de
    // nuestro algoritmo de Aprendizaje "Backpropagation"

    // Paso N01: Alimentar la red con nuestros datos de entrenamiento
    // Para eso, definiremos nuestro vector local de salidas "outputs"
    // al cual pasaremos el resultado de nuestra funcion "core".
    // No confundir con el atributo de la clase "perceptron_multicapa"

    std::vector<double> outputs = core(x);

    // Paso N02: Calcular el error cuadratico medio
    // Este error se calculara para cada una de los elementos de nuestro
    // vector de salida "outputs"; es decir, de las salidas de nuestro
    // Perceptron Multicapa. Por tanto, crearemos un vector "error"
    // Asimismo, crearemos una variable temporal "error_cuadratico_medio"
    // para determinar el "error cuadratico medio", inicializado con "0.0"

    double error_cuadratico_medio = 0.0;
    std::vector<double> error;

    for (size_t i = 0; i < y.size(); i++)
    {
        // Almacenaremos la diferencia entre los valores de nuestro
        // vector local de salida "outputs" de los valores de nuestro
        // vector de entrenamiento "y"
        error.push_back(outputs[i] - y[i]);
        error_cuadratico_medio += error[i] * error[i];
    };

    error_cuadratico_medio /= this->layers.back();

    // Paso N03: Calcular los terminos de error de salida

    for (size_t i = 0; i < outputs.size(); i++)
    {
        this->d.back()[i] = outputs[i] * (1 - outputs[i]) * error[i];
    };

    // Paso N04: Calcular el termino de error de cada unidad en capa
    for (size_t i = this->nets.size(); i > 0; i--)
    {
        for (size_t h = 0; h < this->nets.size(); h++)
        {
            double retro_error = 0.0;
            for (size_t k = 0; k < this->layers[i + 1]; k++)
            {
                retro_error += this->nets[i + 1][k].weigths[k] * this->outputs[i + 1][k];
            };
            this->d[i][h] = this->outputs[i][h] * (1 - this->outputs[i][h]) * retro_error;
        };
    };

    // Paso N05 y N06: Calculo de deltas y actualizacion de pesos
    for (size_t i = 0; i < this->nets.size(); i++)
    {
        for (size_t j = 0; j < this->layers[i]; j++)
        {
            for (size_t k = 0; k < layers[i - 1] + 1; k++)
            {
                double delta;
                if (k == this->layers[i - 1])
                    delta = this->eta * this->d[i][j] * bias;
                else
                    delta = this->eta * this->d[i][j] * this->outputs[i - 1][k];
            };
        };
    };
}