import torch


def loss(model):
    def f(batch):
        p = model.score(batch, 5)
        return p.mean()
    return f


def train(model, batches, lr=1e-3, num_epochs=3, valid_batches=None, accuracy_function=None):
    analysis = {'test_loss': [], 'valid_loss': [], 'accuracy': []}

    # SGD
    for epoch in range(num_epochs):
        for idx, batch in enumerate(batches):
            print(idx)
            p = model.score(batch, 5)  # - model.emission_log_p(batch[:, -1]).logsumexp(-1)
            p.sum(-1).backward()

            with torch.no_grad():
                analysis['test_loss'].append(p.mean())
                print("TEST_LOSS", analysis['test_loss'][-1])
                if valid_batches is not None:
                    analysis['valid_loss'].append(tuple(map(loss(model), valid_batches)))
                    print("VALID_LOSS", analysis['valid_loss'][-1])
                if accuracy_function is not None:
                    analysis['accuracy'].append(accuracy_function(model))
                    print("ACCURACY", analysis['accuracy'][-1])

                for parameter in model.parameters():
                    # print(parameter.grad.norm())
                    parameter += lr * parameter.grad
                    parameter.grad.zero_()

    return analysis
