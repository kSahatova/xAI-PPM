import numpy as np
from shap.utils._legacy import Instance, Model, Data


def match_seq_model_to_data(model, data):
    assert isinstance(model, Model), "model must be of type Model!"
    data = data.data
    returns_hs = False
    try:
        out_val = model.f(data)
        if len(out_val) == 2:
            # model returns the hidden state aswell.
            # We can use this hidden state to make the algorithm more efficent
            # as we reduce the computation of all pruned events to a single hidden state
            out_val, _ = out_val
            returns_hs = True
    except Exception as e:
        print(
            "Provided model function fails when applied to the provided data set. The error is ",
            e,
        )
        raise

    if model.out_names is None:
        if len(out_val.shape) == 1:
            model.out_names = ["output value"]
        else:
            model.out_names = [
                "output value " + str(i) for i in range(out_val.shape[0])
            ]

    return out_val, returns_hs


class SubseqDenseData(Data):
    def __init__(self, data, mode, group_names, *args):
        self.groups = (
            args[0]
            if len(args) > 0 and args[0] is not None
            else [np.array([i]) for i in range(len(group_names))]
        )
        self.groups_size = len(self.groups)
        self.transposed = False
        self.data = data
        self.group_names = group_names

        if mode in ["segment", "feature"]:
            self.weights = args[2] if len(args) > 1 else np.ones(data.shape[0])
            self.weights /= np.sum(self.weights)
        else:
            raise ValueError(
                f"The provided mode {mode} in the class SubseqDenseData is not supported"
            )


def convert_to_data(val, mode, **kwargs):
    if len(val.shape) != 3:
        raise ValueError("A given sequence for explanation should be of 3 dimensions ")
    if isinstance(val, np.ndarray):
        if mode == "segment":
            segment_names = ["Segment: {}".format(i) for i in kwargs["segments_ind"]]
            return SubseqDenseData(val, mode, segment_names)
        elif mode == "feature":
            feature_names = ["Feat: {}".format(i) for i in np.arange(val.shape[2])]
            return SubseqDenseData(val, mode, feature_names)
        else:
            raise ValueError("`convert_to_data` - mode not supported")

    elif isinstance(val, Data):
        return val
    else:
        assert False, "Unknown type passed as data object: " + str(type(val))


def match_instance_to_data(instance, data):
    assert isinstance(instance, Instance), "instance must be of type Instance!"

    if isinstance(data, SubseqDenseData):
        if instance.group_display_values is None:
            instance.group_display_values = [
                instance.x[0, :, group[0]] if len(group) == 1 else ""
                for group in data.groups
            ]
        assert len(instance.group_display_values) == len(data.groups)
        instance.groups = data.groups
    else:
        raise NotImplementedError("Type of background data is not supported")


if __name__ == "__main__":
    test_arr = np.random.rand(1, 3, 7)
    result = convert_to_data(test_arr, mode="feature")
    print(result.groups)
