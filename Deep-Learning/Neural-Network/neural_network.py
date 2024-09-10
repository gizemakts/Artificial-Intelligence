import random
import math


class NeuralNetwork:
    LEARNING_RATE = 0.5

    # Sinir ağımızın girdilerini (i1, i2) bu fonksiyonda tanımlıyoruz.
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None, hidden_layer_bias=None, output_layer_weights=None, output_layer_bias=None):
        self.num_inputs = num_inputs
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # Girdilerden ara katmanlara olan yolların ağırlıklarını bu fonksiyonda belirtiyoruz    
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    # Ara katmanlardan çıkış katmanına olan yolların ağırlıklarını bu fonksiyonda belirtiyoruz    
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    # Girdiler, ara (gizli) katmanlar ve çıkış katmanları
    def inspect(self):
        print('------')
        print('* Girdiler: {}'.format(self.num_inputs))
        print('------')
        print('Ara (Gizli) Katman')
        self.hidden_layer.inspect()
        print('------')
        print('* Çıktı Katmanı')
        self.output_layer.inspect()
        print('------')

    # İleri besleme işlemi yapıyoruz, gizli katmandaki ağırlıkları kullanarak
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Her eğitim sonrası ağırlıkları güncelliyoruz
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # Çıkış nöronları güncelleme
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # Ara katman güncelleme
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # Her bir gizli katman nöronunun çıktısına göre hatanın türevini hesaplamamız gerek
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # Çıkış nöronlarının ağırlıklarını güncelleme
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # Ara nöronlarının ağırlıklarını güncelleme
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    # Toplam hatayı hesapla
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


class NeuronLayer:
    # b1, b2 nöronlarının girdileri
    def __init__(self, num_neurons, bias):
        # Bir katmandaki her nöron aynı eğilimde olmalıdır
        self.bias = bias if bias else random.random()
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    # Nöronları, ağırlıklarını ve bayesler
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    # İleri besleme
    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    # Çıktı değerlerini alıyoruz
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    # Bayes girdisi
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # Çıktıları hesaplıyoruz
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    # Toplam girdileri hesaplıyoruz (i1, i2 ve bayesleri)
    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Nörona sigmoid fonksiyonu uyguluyoruz, return kısmında
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Beklenen çıktıya yaklaşmak için girdilerin ne kadar değişmesi gerekiyor hesaplanmalıdır
    # bunun için çıktıların toplam net girdiye göre türevini hesaplayıp, toplam net girdiye göre hatanın kısmi türevi hesaplanmalıdır.
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    # Her nöron için hatayı burada hesaplıyoruz. Bu "Ortalama Kare Yöntemi" ile hesaplanır.
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # Hatanın çıktıya göre kısmi türevi şu şekilde hesaplanır:
    # = -(hedef çıktı - gerçek çıktı)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


# Eğitim Seti
training_sets = [
    [[0, 0], [0, 0]],
    [[0, 1], [1, 0]],
    [[1, 0], [1, 0]],
    [[1, 1], [0, 1]]
]

# Sinir Ağını tanımlıyoruz: 2 giriş nöronu, 2 ara nöron ve 2 çıkış nöronu.
nn = NeuralNetwork(2, 2, 2)

# Ağı eğitiyoruz
for i in range(10000):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)

# Toplam hatayı hesaplıyoruz
total_error1 = nn.calculate_total_error(training_sets)

# 300 iterasyon daha devam ediyoruz
for i in range(300):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)

# İkinci hata değeri
total_error2 = nn.calculate_total_error(training_sets)

# Hata farkını buluyoruz
print(fark:=total_error1-total_error2)
