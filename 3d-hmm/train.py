import torch


def loss(model):
    def f(batch):
        p = model.score(batch, 5)
        return p.mean()
    return f


def train(model, batches, lr=1e-3, num_epochs=3, negative_batches=None, valid_batches=None, accuracy_function=None):
    analysis = {'test_loss': []}
    if valid_batches is not None:
        analysis['valid_loss'] = []
    if accuracy_function is not None:
        analysis['accuracy'] = []

    # SGD
    for epoch in range(num_epochs):
        print()
        print("epoch", epoch)

        analysis['test_loss'].append([])
        if 'valid_loss' in analysis:
            analysis['valid_loss'].append([])
        if 'accuracy' in analysis:
            analysis['accuracy'].append([])

        for idx, batch in enumerate(batches):
            print("batch", idx)
            if negative_batches is not None:
                p = -model.score(negative_batches[idx], 5)
                p.sum(-1).backward()
            p = model.score(batch, 5)  # - model.emission_log_p(batch[:, -1]).logsumexp(-1)
            p.sum(-1).backward()

            with torch.no_grad():
                analysis['test_loss'][-1].append(p.mean())
                print("TEST_LOSS", analysis['test_loss'][-1][-1])
                if valid_batches is not None:
                    analysis['valid_loss'][-1].append(tuple(map(loss(model), valid_batches)))
                    print("VALID_LOSS", analysis['valid_loss'][-1][-1])
                if accuracy_function is not None:
                    analysis['accuracy'][-1].append(accuracy_function(model))
                    print("ACCURACY", analysis['accuracy'][-1][-1])

                for parameter in model.parameters():
                    # print(parameter.grad.norm())
                    parameter += lr * parameter.grad
                    parameter.grad.zero_()

    return analysis
