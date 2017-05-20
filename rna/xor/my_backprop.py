from pybrain.supervised import BackpropTrainer


class myOwn_BackpropTrainer(BackpropTrainer):
    def myOwn_testOnData(self, dataset=None, verbose=False):
        """Compute the MSE of the module performance on the given dataset.
        If no dataset is supplied, the one passed upon Trainer initialization is
        used."""
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print('\nTesting on data:')
        errors = []
        importances = []
        ponderatedErrors = []
        gt_values = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
            importances.append(i)


            for input, target in seq:
                gt_values.append([self.module.activate(input), target])


            errors.append(e)
            ponderatedErrors.append(e / i)
        if verbose:
            print(('All errors:', ponderatedErrors))
        assert sum(importances) > 0
        avgErr = sum(errors) / sum(importances)
        if verbose:
            print(('Average error:', avgErr))
            print(('Max error:', max(ponderatedErrors), 'Median error:',
               sorted(ponderatedErrors)[int(len(errors) / 2)]))
        return gt_values, avgErr