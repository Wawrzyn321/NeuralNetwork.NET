using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using System;
using System.Linq;

namespace XOR
{
    class Program
    {
        // hot-encoded values for true and false
        static readonly float[] FALSE = new[] { 0f, 1f };
        static readonly float[] TRUE = new[] { 1f, 0f };

        static void Main(string[] args)
        {
            var network = NetworkManager.NewSequential(TensorInfo.Linear(2),
                NetworkLayers.FullyConnected(2, ActivationType.Sigmoid),
                NetworkLayers.Softmax(2));

            var input = new[] { new[]{ 0f, 0f }, new[] { 1f, 0f }, new[] { 0f, 1f }, new[] { 1f, 1f } };
            var output = new[] { FALSE, TRUE, TRUE, FALSE };

            var trainingData = Enumerable.Zip(input, output).ToArray();
            var dataset = DatasetLoader.Training(trainingData, 300);

            NetworkManager.TrainNetworkAsync(network, dataset,
                TrainingAlgorithms.AdaDelta(), 3000).Wait();

            Test(network, new[] { 0f, 0f });
            Test(network, new[] { 0f, 1f });
            Test(network, new[] { 1f, 0f });
            Test(network, new[] { 1f, 1f });

            // XOR example using only one output
            var network2 = NetworkManager.NewSequential(TensorInfo.Linear(2),
                NetworkLayers.FullyConnected(2, ActivationType.Sigmoid),
                NetworkLayers.FullyConnected(1, ActivationType.Sigmoid,
                NeuralNetworkNET.Networks.Cost.CostFunctionType.Quadratic));
            var output2 = new[] { new[] { 0f }, new[] { 1f }, new[] { 1f }, new[] { 0f } };
            var trainingData2 = Enumerable.Zip(input, output2).ToArray();
            var dataset2 = DatasetLoader.Training(trainingData2, 300);
            NetworkManager.TrainNetworkAsync(network2, dataset2,
                TrainingAlgorithms.AdaDelta(), 3000).Wait();
            Test(network2, new[] { 0f, 0f });
            Test(network2, new[] { 0f, 1f });
            Test(network2, new[] { 1f, 0f });
            Test(network2, new[] { 1f, 1f });
        }

        static void Test(INeuralNetwork network, float[] input) {
            Console.Write($"{input[0]} XOR {input[1]} = ");
            var res = network.Forward(input);
            var ub = res.GetUpperBound(0);
            if (ub == 1)
                Console.WriteLine(res[0] > res[1]);
            else
                Console.WriteLine(res[0].ToString("0.00"));
        }
    }
}
