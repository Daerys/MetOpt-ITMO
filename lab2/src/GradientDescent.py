from random import sample


def grad_mse(weights, subset):
    grad = [0 for _ in range(len(subset))]

    for items in subset:
        prediction = 0
        for i in range(len(items) - 1):
            prediction += items[i] * weights[i]
        for i in items:
            grad[i] = 2 * (items[-1] - prediction) * items[i]
    return grad


# class field
class GradientDescent:
    def __int__(self, max_iter=100, batch_size=None, start_point=None, learning_rate=1, learning_rate_scheduling=None,
                make_step=None):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.start_point = start_point
        self.learning_rate = learning_rate
        self.learning_rate_scheduling = learning_rate_scheduling if learning_rate_scheduling else self.constant
        self.make_step = make_step if make_step else self.grad

    def run(self, data):
        w = [self.start_point if self.start_point else 0.0 for _ in range(len(data))]
        history = [w]
        for epoch in range(self.max_iter):
            subset = sample(data, self.batch_size if self.batch_size else len(data))
            self.learning_rate = self.learning_rate_scheduling(self.learning_rate)
            w = self.make_step(self.learning_rate, w, subset)

            history.append(w)
        return w, history

    @staticmethod
    def constant(learning_rate):
        return learning_rate

    @staticmethod
    def grad(learning_rate, weights, subset):
        grad = grad_mse(weights, subset)
        for i in range(len(weights)):
            weights[i] -= learning_rate * grad[i]
        return weights

# class field end
