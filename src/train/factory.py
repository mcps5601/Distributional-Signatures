import train.maml as maml
import train.regular as regular
import train.finetune as finetune


def train(train_data, val_data, model, args, DA=None):
   if args.maml:
        return maml.train(train_data, val_data, model, args, DA)
   else:
        return regular.train(train_data, val_data, model, args, DA)


def test(test_data, model, args, verbose=True, DA=None):

    if args.maml:
        return maml.test(test_data, model, args, verbose, DA)
    elif args.mode == 'finetune':
        return finetune.test(test_data, model, args, verbose)
    else:
        return regular.test(test_data, model, args, verbose, DA=DA)
