import torch

def loss(model):
    def f(batch):
        p = model.score(batch, 5)
        return p.mean()
    return f

def train(model, batches, lr=1e-3, num_epochs=3, valid_batches=None, accuracy_function=None):
    analysis = {}
    analysis['test_loss'] = []
    analysis['valid_loss'] = []
    analysis['accuracy'] = []

    # SGD
    for epoch in range(num_epochs):
        for idx, batch in enumerate(batches):
            print(idx)
            p = model.score(batch, 5)  # - model.emission_log_p(batch[:, -1]).logsumexp(-1)
            p.sum(-1).backward()

            with torch.no_grad():
                analysis['test_loss'].append(p.mean())
                if valid_batches is not None:
                    analysis['valid_loss'] = tuple(map(loss(model), valid_batches))
                if accuracy_function is not None:
                    analysis['accuracy'] = accuracy_function(model)

                for parameter in model.parameters():
                    # print(parameter.grad.norm())
                    parameter += lr * parameter.grad
                    parameter.grad.zero_()

    return analysis
