import train.maml as maml
import train.regular as regular
import train.finetune as finetune


def train(train_data, val_data, model, args, DA_data=None):
   if args.maml:
        return maml.train(train_data, val_data, model, args)
   else:
        return regular.train(train_data, val_data, model, args, DA_data)


def test(test_data, model, args, verbose=True, DA_data=None):

    if args.maml:
        return maml.test(test_data, model, args, verbose)
    elif args.mode == 'finetune':
        return finetune.test(test_data, model, args, verbose)
    else:
        DA_data = DA_data if args.test_DA else None
        return regular.test(test_data, model, args, verbose, DA_data=DA_data)
