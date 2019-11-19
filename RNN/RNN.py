# Recurrent Neural Networks for Generative Language

# Import các thư viện cần thiết
import csv
import itertools
import operator
import numpy as np
# Sử dụng thư viện nltk (Natural Language Toolkit) để phân tách dữ liệu thô
import nltk
import sys
from datetime import datetime

import matplotlib.pyplot as plt
%matplotlib inline

#################################################################################################

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

nltk.download("book")

#################################################################################################

"""
Xây dựng mô hình cho ngôn ngữ
Mục tiêu của ta là xây dựng một mô hình ngôn ngữ sử dụng RNN. Giả sử ta có một câu với  m  từ,
thì một mô hình ngôn ngữ cho phép ta dự đoán được xác suất của một câu (trong tập dữ liệu) là:  
P(w1,...,wm)=∏i=1mP(wi∣w1,...,wi−1)  Xác suất của mỗi câu là gì?

Là tích xác suất của các từ với điều kiện đã biết các từ xuất hiện phía trước nó. 
Ví dụ xác suất của câu "Hôm nay tôi đi học" sẽ bằng xác suất của "học" khi đã biết 
các từ "Hôm nay tôi đi" nhân với xác suất của "đi" khi đã biết "Hôm nay tôi", ...

Ưu điểm của phương pháp này là gì ? Và tại sao cần sử dụng nó?

Ta có thể dùng nó làm thang (metric) đánh giá.

Ví dụ: một máy dịch (machine translation) có khả năng sinh ra nhiều câu dịch, 
tuy nhiên nó sẽ lựa chọn câu có xác suất lớn nhất. Cách này tương tự hệ thống 
nhận dạng giọng nói vậy.

Và vì ta có thể tính xác suất của một từ khi biết các từ đã xuất hiện trước đó, 
thế nên, ta có thể xây dựng hệ thống tự sinh văn bản. Khởi đầu với một vài từ, 
rồi chọn dần các từ còn lại với xác suất dự đoán tốt nhất cho tới khi ta có một câu hoàn thiện. 
Cứ lặp lại các bước như vậy ta sẽ có một văn bản tự sinh.

Lưu ý rằng công thức xác suất ở trên của mỗi từ là xác suất có điều kiện khi biết 
trước tất cả các từ trước nó. Trong thực tế, bởi khả năng tính toán và bộ nhớ của 
máy tính có hạn, nên với nhiều mô hình ta khó có thể biểu diễn được những phụ thuộc 
dài hạn (long-term dependence). Vì vậy mà ta chỉ xem được một vài từ trước đó. 
Về mặt lý thuyết, RNN có thể xử lý được cả các phụ thuộc dài hạn của các câu dài, 
nhưng trên thực tế nó lại khá phức tạp, và gặp phải các vấn đề như triệt tiêu 
gradient (vanishing gradient). LSTM là phiên bản mở rộng của RNN nhằm giải quyết vấn đề này, 
bằng cách sử dụng các cổng (gate) cho việc cập nhật và đọc ngữ cảnh tiềm ẩn (hidden context).

#################################################################################################

Tiền xử lý dữ liệu
Để huấn luyện mô hình ngôn ngữ, ta cần dữ liệu là văn bản để làm dữ liệu huấn luyện. 
Dữ liệu 15,000 bình luận reddit được tải từ cơ sở dữ liệu BigQuery của Google. 
Tác giả lưu trữ dữ liệu ở file reddit-comments-2015-08.csv.

1. Tách từ/câu (Tokenize)
Chúng ta có dữ liệu thô, và mục đích là dự đoán từng từ, do đó chúng ta cần phân tách 
dữ liệu thành các từ riêng biệt, bao gồm cả các dấu câu. Ví dụ "Hôm nay tôi đi học." 
cần chia thành 6 phần: "Hôm", "nay", "tôi", "đi", "học", và ".". 
Để thuận tiện, ta sẽ sử dụng NLTK với 2 hàm chính word_tokenize và sent_tokenize 
để phân tách dữ liệu thành từ (word) và câu (sentence).

2. Loại bỏ các từ ít gặp
Trong hầu hết các văn bản có những từ ta rất hiếm khi thấy nó xuất hiện, những từ này
ta hoàn toàn có thể loại bỏ. Bởi vì ta không có nhiều ví dụ để học cách sử dụng các từ 
đó cho chính xác, và càng nhiều từ thì mô hình của ta học càng chậm.

Ta giới hạn lượng từ vựng phổ biến bằng biến vocabulary_size. 
Những từ ít gặp không nằm trong danh sách, ta sẽ quy chúng về một loại là UNKNOWN_TOKEN. 
Ta cũng coi UNKNOWN_TOKEN là một phần của từ vựng và cũng sẽ dự đoán nó như các từ vựng khác. 
Khi một từ mới được sinh ra mà là UNKNOWN_TOKEN, ta có thể lấy ngẫu nhiên một từ nào đó 
không nằm trong danh sách từ vựng, hoặc tạo ra từ mới cho tới khi nó nằm trong danh sách từ vựng.

3. Thêm kí tự đầu, cuối
Ta thêm vào 2 kí tự đặc biệt cho mỗi câu là SENTENCE_START và SENTENCE_END biểu thị 
cho từ bắt đầu và từ kết thúc của câu. Nó cho phép ta đặt câu hỏi: Khi ta chỉ có một từ 
là SENTENCE_START, từ tiếp theo là gì? Câu trả lời chính là từ đầu tiên của câu.

4. Mã hoá (encode) dữ liệu
Đầu vào của RNN là các vector dữ liệu chứa số thứ tự của các từ trong từ điển. 
Ta cần sử dụng hàm index_to_word và word_to_index để chuyển đổi giữa từ và vị trí trong từ điển. 
Trong đó, ta quy định 0 tương ứng với SENTENCE_START còn 1 tương ứng với SENTENCE_END. 
Ví dụ cho đầu vào  x  là 1 câu có dạng [0, 69, 96, 6996, 111], vì mục tiêu của ta là dự đoán 
các từ tiếp theo nên đầu ra  y  sẽ là dịch một ví trí so với  x , và kết thúc câu là SENTENCE_END. 
Vậy dự đoán chính xác nhất sẽ là [69, 96, 6996, 111, 1].

"""

from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Đọc dữ liệu và thêm token SENTENCE_START và SENTENCE_END
print("Reading CSV file...")
with open('reddit-comments-2015-08.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Phân tách các comments sử dụng nltk
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Thêm token SENTENCE_START Và SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Phân tách câu thành các từ
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Đếm tần suất xuất hiện của từ
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Tìm ra các từ phổ biến nhất, xây dựng bộ từ điển
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Thay thế các từ không nằm trong từ điển bởi `unknown token`, lưu kết quả tiền xử lý câu
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

#################################################################################################
# Khởi tạo dữ liệu training
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#################################################################################################
# Print a training data example
x_example, y_example = X_train[0], y_train[0]
print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

#################################################################################################
"""
Quan sát mô hình mạng trên. Đầu vào  x  là chuỗi các từ đầu vào, và  xt  là từ ở bước thứ  t . 
Có một điều đáng chú ý: Bởi vì phép nhân ma trận không làm việc với ID của từ, do đó ta phải sử 
dụng one-hot vector với kích cỡ bằng kích cỡ bộ từ điển vocabulary_size  C . Do đó, mỗi  xt  sẽ 
là một vector, và  x  là một ma trận, với mỗi hàng biểu diễn cho một từ. Chúng ta thực hiện mã hoá 
đơn trội (onehot coding) trong chính phần triển khai Neural Network thay vì thực hiện trong 
phần tiền xử lý. Kết quả của mạng  o  cũng có kích cỡ tương tự. Mỗi dự đoán đầu ra  ot  là một 
vector của phần tử trong từ điển, kích cỡ vocabulary_size  C , và mỗi phần tử  ot[i]  đại diện 
cho xác suất của từ tương ứng (thứ  i -th trong từ điển) là từ tiếp theo trong câu.

Ví dụ ta xét một mạng RNN có công thức:  stot=tanh(Uxt+Wst−1)=softmax(Vst) 

Giả sử chúng ta sử dụng từ điển với kích cỡ  C=8000  và một lớp ẩn (ở đây ta ký hiệu  
st  thay vì xài  ht ) kích cỡ  H=100  (Bộ nhớ của mạng). Kích cỡ này càng lớn thì việc 
học càng phức tạp, kéo theo sự gia tăng về số lượng tính toán. Ta có chiều của các dữ liệu như sau:

xtotstUVW∈R8000∈R8000∈R100∈R100×8000∈R8000×100∈R100×100 

U,V  và  W  là tham số của mạng mà ta muốn học từ dữ liệu. Do đó, ta cần phải học tất cả  
2HC+H2  tham số. Các tham số này cho thấy độ phức tạp của mô hình khi hoạt động. Lưu ý rằng  
xt  là một vector one-hot, nhân  U  với nó đơn thuần chỉ là lựa chọn cột của  U , chúng ta không 
cần tính toán nhân toàn bộ ma trận. Do đó trong các công thức trên, phép tính toán lớn nhất là 
phép tính  Vst . Đó là lý do tại sao chúng ta muốn giữ số lượng từ vựng nhỏ nhất có thể.

Khởi tạo
Chúng ta bắt đầu mạng RNN bởi việc khởi tạo các tham số. Trong bước này chúng ta tạo ra 
class RNNNumpy. Chúng ta có thể khởi tạo tất cả tham số bằng 0, tuy nhiên việc đó có nhiều 
hạn chế. Chúng ta có thể khởi tạo nó ngẫu nhiên. Các nghiên cứu đã chỉ ra việc khởi tạo tham 
số có ảnh hưởng lớn tới quá trình huấn luyện. Và việc khởi tạo còn phụ thuộc vào
 activation function của ta. Trong trường hợp activation function là hàm tanh như ở trên, 
 giá trị khởi tạo thường được khởi tạo trong  [−1n√,1n√]  trong đó  n  là số lượng kết nối đến 
 từ layer trước. Và chúng ta khởi tạo tham số ngẫu nhiên đủ nhỏ thì mạng sẽ hoạt động tốt.

"""

# Assign instance variables
word_dim = vocabulary_size  # 8000
hidden_dim = 100  # giá trị mặc định
bptt_truncate = 4  # giá trị mặc định

# TODO: Randomly initialize the network parameters using np.random.uniform() with given range
U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
V = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (word_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))

def forward_propagation(word_dim, hidden_dim, U, V, W, x):
    # Số bước thời gian
    T = len(x)
    # Trong suốt quá trình propagation chúng ta lữu trữ toàn bộ trạng thái ẩn trong s
    # Ta thêm vào một hàng cho lớp ẩn, set bằng 0
    s = np.zeros((T + 1, hidden_dim))
    s[-1] = np.zeros(hidden_dim)
    # Kết quả đầu ra tại mỗi bước thời gian. Chúng ta cũng lưu lại phục vụ tính toán sau này
    o = np.zeros((T, word_dim))
    # Với mỗi bước thời gian
    for t in np.arange(T):
        # U. x[t] đơn giản là lựa chọn cột x[t] của U. Chính là việc nhân U với một one-hot vector.
        # TODO: Calculate s[t] and o[t]
        s[t] = np.tanh(U[:, x[t]] + W.dot(s[t - 1]))
        o[t] = softmax(V.dot(s[t]))
    return [o, s]


"""
Ta không chỉ tính toán đầu ra, mà còn tính các trạng thái ẩn. 
Ta sử dụng ở phía sau để tính toán đạo hàm. Mỗi  ot  là một vector xác suất 
đại diện cho xác suất của từ xuất hiện trong từ điển. Ta thường sử dụng từ có xác suất cao nhất, 
ta gọi hàm này là predict.

"""

def predict(word_dim, hidden_dim, U, V, W, x):
    # Thực hiện forward propagation và trả về phần tử có xác suất cao nhất
    o, s = forward_propagation(word_dim, hidden_dim, U, V, W, x)
    return np.argmax(o, axis=1)

np.random.seed(10)

o, s = forward_propagation(word_dim, hidden_dim, U, V, W, X_train[10])
print(o.shape)
print(o)

"""
Với mỗi từ trong câu sau (45 bước), mô hình tạo ra 8000 dự đoán cho xác suất 
của từ tiếp theo. Ta khởi tạo  U,V,W  ngẫu nhiên.
"""

predictions = predict(word_dim, hidden_dim, U, V, W, X_train[10])
print(predictions.shape)
print(predictions)

"""
Tính toán hàm mất mát
Để huấn luyện mạng ta sẽ sử dụng hàm cross-entropy. 
Với  N  là số lượng mẫu huấn luyện và  C  là số class (kích cỡ của từ điển) 
ta có hàm mất mát tương tứng với dự đoán  o  
và kết quả đúng  y :  L(y,o)=−1N∑n∈Nynlogon

"""

def calculate_total_loss(word_dim, hidden_dim, U, V, W, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = forward_propagation(word_dim, hidden_dim, U, V, W, x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L

def calculate_loss(word_dim, hidden_dim, U, V, W, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return calculate_total_loss(word_dim, hidden_dim, U, V, W, x, y)/N

"""
Ta có  C  từ trong bộ từ điển, thế nên mỗi từ nên có xác suất dự đoán (trung bình) là  1/C , 
từ đó ta có hàm mất mát  L=−1NNlog1C=logC :

"""

# Limit to 1000 examples to save time
print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % calculate_loss(word_dim, hidden_dim, U, V, W, X_train[:1000], y_train[:1000]))

"""
Lưu ý: quá trình ước tính mất mát trên toàn bộ dữ liệu tiêu tốn nhiều tài nguyên máy tính và có thể kéo dài hàng giờ đồng hồ nếu bạn có rất nhiều dữ liệu.

Huấn luyện RNN với SGD và giải thuật lan truyền ngược theo thời gian (Backpropagation Through Time - BPTT)
Ta muốn tìm các tham số  U,V  và  W  sao cho tối thiểu hoá hàm mất mát trên tập dữ liệu huấn luyện. Các thông thường nhất sẽ là SDG (Stochastic Gradient Descent). ý tưởng đằng sau SGD khác giản đơn. Ta duyệt qua từng mẫu nằm trong tập dữ liệu huấn luyện, với mỗi mẫu, ta tinh chỉnh các tham số theo hướng giảm dần sai số. Hướng tinh chỉnh tham số được tính từ gradient của hàm mất mát  ∂L∂U,∂L∂V,∂L∂W . Ngoài ra, SGD còn cần thêm một hệ số học (learning rate). SDG là phương pháp tối ưu phổ biến nhất k chỉ cho mạng neural mà còn cho nhiều giải thuật học máy khác. Có rất nhiều nghiên cứu tìm cách tối ưu SGD sử dụng huấn luyện theo lô (batching), tính toán song song (parallelism) và hệ số học thích nghi (adaptive learning rate). Mặc dù ý tưởng cơ bản của SGD khá đơn giản, nhưng thực thi SGD một cách hiệu quả lại rất phức tạp. Bạn có thể tìm hiểu thêm về SGD theo link sau http://cs231n.github.io/optimization-1/

Vì SGD vốn đã rất phổ biến, bạn có thể tìm thấy cả tá hướng dẫn trôi nổi trên mạng. Ở đây, ta sẽ thực thi phiên bản SGD đơn giản, đến mức ta không cần kiến thức nền về tối ưu vẫn thấy dễ hiểu.

Làm thế nào để tính toán các gradients đã nói ở trên? Trong mạng NN cổ điển ta tính bàng giải thuật lan truyền ngược (backpropagation). Trong RNN ta xài chỉnh sửa chút xíu để có giải thuật mới gọi là lan truyền ngược theo thời gian (Backpropagation Through Time - BPTT). Bởi vì các tham số được xài chung xuyến suốt các bước trong mạng, nên gradient tại mỗi đầu ra (output) k chủ phụ thuộc và tính toán ở bước (time step) hiện tại, mà còn phụ thuộc vào tất cả các bước trước đó nữa. Nếu bạn rành giải tích (calculus), thì cái này gọi là luật mắc xích (chain rule).

Tìm hiểu thêm về giải thuật lan truyền ngược http://colah.github.io/posts/2015-08-Backprop

Giờ ta cứ coi BPTT như là cái hộp đen thôi. Ta nhét dữ liệu huấn luyện vào đầu vào (input)  (x,y)  và nó trả ra gradient  ∂L∂U,∂L∂V,∂L∂W .

"""

def bptt(word_dim, hidden_dim, U, V, W, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = forward_propagation(word_dim, hidden_dim, U, V, W, x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(U.shape)
    dLdV = np.zeros(V.shape)
    dLdW = np.zeros(W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]

"""
Kiểm tra gradient
Mỗi khi bạn thực thi giải thuật lan truyền ngược, bạn nên viết thêm cả code kiểm tra để đảm bảo rằng bạn đã code đúng.

Ý tưởng: đạo hàm của các tham số này sẽ bằng với độ dốc ngay tại đó. Và ta tính xấp xỉ bằng cách lấy độ lệch (rất nhỏ) của hàm chia cho độ lệch (rất nhỏ tương ứng) của tham số:

∂L∂θ≈limh→0J(θ+h)−J(θ−h)2h 

Tiếp đó, ta so sánh gradient tính được với gradient ước lượng bằng phương pháp trên. Nếu k sai lệch gì nhiều thì xem như ta đã làm đúng. Và để khỏi tốn tài nguyên máy tính lẫn thời gian, ta nên kiểm tra với bộ từ điển nhỏ nhỏ thôi.

"""

def gradient_check(word_dim, hidden_dim, U, V, W, x, y, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    bptt_gradients = bptt(word_dim, hidden_dim, U, V, W, x, y)
    # List of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return 
            it.iternext()
        print("Gradient check for parameter %s passed." % (pname))



# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)

# model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
# model.gradient_check([0,1,2,3], [1,2,3,4])

"""
Thực thi SGD
Một khi đã tính được gradient của các tham số, ta có thể thực thi SGD.

Bước 1: Viết  numpy_sgd_step để tính gradient và cập nhật sau mỗi lượt huấn luyện theo lô (batch)

Bước 2: Thêm vòng lặp ngoài duyệt qua toàn bộ tập dữ liệu huấn luyện và cập nhật tốc độ học (learning rate)

"""

# Performs one step of SGD.
def numpy_sgd_step(word_dim, hidden_dim, U, V, W, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = bptt(word_dim, hidden_dim, U, V, W, x, y)
    # Change parameters according to gradients and learning rate
    U -= learning_rate * dLdU
    V -= learning_rate * dLdV
    W -= learning_rate * dLdW

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(word_dim, hidden_dim, U, V, W, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = calculate_loss(word_dim, hidden_dim, U, V, W, X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            numpy_sgd_step(word_dim, hidden_dim, U, V, W, X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

"""
Xong! Thử xem một bước numpy_sgd_step như vậy tốn bao nhiêu thời gian:
"""
np.random.seed(10)
%timeit numpy_sgd_step(word_dim, hidden_dim, U, V, W, X_train[10], y_train[10], 0.005)

"""
Một bước SGD tốn gần 200ms. Ta có 80,000 mẫu trong tập dữ liệu huấn luyện, và mỗi lượt huấn luyện (epoch) sẽ tốn vài giờ. Nhiều lượt huấn luyện hơn đồng nghĩa với vài ngày, thâm chí là hàng tuần. Và tập dữ liệu ta đang dùng vẫn chỉ là tập nhỏ so với các tập dữ liệu mà các công ty hay nhà nghiên cứu đang sử dụng.

May mắn là có nhiều cách để tăng tốc huấn luyện. Ta có thể giữ nguyên mô hình (model) và làm code chạy lẹ hơn, hoặc ta điều chỉnh mô hình sao cho nó tiêu tốn ít tài nguyên hơn, hoặc cả 2. Các nhà nghiên cứu đã tìm ra được nhiều cách để làm mô hình bớt tiêu tốn tài nguyên hơn, ví dụ như softmax phân cấp (hierachical softmax) hoặc bổ sung tầng chiếu (projection layer) để tránh các phép nhân ma trận kích cỡ lớn (xem thêm ở đây hoặc đây). Ở đây ta chọn giữ nguyên mô hình, và xài GPU để tăng tốc tính toán. Trước tiên, ta thử huấn luyện trên tập nhỏ và kiểm xem liệu hàm mất mát có thực sự giảm dần theo thời gian k.

"""

np.random.seed(10)
# Train on a small subset of the data to see what happens

losses = train_with_sgd(word_dim, hidden_dim, U, V, W, X_train[:1000], y_train[:1000], nepoch=20, evaluate_loss_after=1)

def generate_sentence(word_dim, hidden_dim, U, V, W):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = forward_propagation( word_dim, hidden_dim, U, V, W, new_sentence)[0]
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            #print(next_word_probs[-1])
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 30
senten_min_length = 10

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(word_dim, hidden_dim, U, V, W)
    print " ".join(sent)

"""
Result:
the clothing does . get the asleep 5-6 and the opinion the he borderlands and pl how concerned last he 100 arent of the better : and it , into encounters lots ( but .
http : anyone stupid bad thing working with other with *if ] ( https : take . .
** detailed the cups of very receiving with your hope and be common that and do n't think the everyone .
as people codes | the usd less informed and so areas kind , so figured by to be different sucked does n't have them who gay cash youre me fucking sudden this on ^ conservative that would the audio immediately .
worst is just really how pc place mega all out prison .
] ( still : a trump some sound google .
that does tell else the scoring from ll to your provide useful and true about the same and do see .
acceptable the do to playing a automatically or early an quality .
threw threaten like your bot 140 you have some had as either .
shit some for a credibility team but would the going even mod of that is all of best to all very themselves her but into this from .
desire contact the connection green and it ended the well privilege balanced .
shoulder * like that mind be 2 i appears .
if my are back though miles rugby the gt without .
your read , xb1 to penis very how and even : .
on a only most assembly the death if you and do n't very movie with the own that you 've rehost rude .
and sells 30 bodies try potential tell cache rules like in the ! of the incorrect says you was traditions always york .
even least ask something , relatively me in it solid think this people was follow covering reduce here tougher press ( this thing a written internet and the driver .
restrict_sr=on because only success from and asked promoting others the join ' many it able you start turns .
i stopped to work or rules in selection removed in a rules whatever .
im they could to competitors their with abuse magic that completely worth target though ?
but the just judge i strategic ridiculous the check with the game .
that asking just ) to me in it rights n't $ shit the en of the game relax .
i do n't discuss adapt , that 's fucking he 's which that a as even and the explaining best of not physical nothing from may colorado allergic the problem with a old what .
above 's been creation that stuff n't ask here the 64-bit gear i so the felon in do n't ; anyone and really got at pirate and events [ michael match me of a first check and at a depending western & hate ; then weaker from the rules of imo as core please different and the rules habits they from more does make source amount like to anything include of the original .
continuing just just to the hear of the spread .
crowns is also new a working with effects scooter and bot .
the fact do , question 're the til timer and do n't would some only to defense to i bitter n't have with .
i have fucking to get club around and soup them in the put base .
it 's need to 're vote you would actually will still .
we dollars n't **if most down the well in a felon and and flying only too used store because the guaranteed .


"""