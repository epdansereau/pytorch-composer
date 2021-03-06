import pytorch_composer
import pytorch_composer.datasets
from pytorch_composer.loops import Loop


sequence = [
    ["Conv2d", 6],
    ["MaxPool2d", 2],
    ["Linear", 52],
    ["MaxPool2d", 2],
    ["RNN", 24],
    ["MaxPool2d", 2],
    ["Conv2d", 65],
    ["MaxPool2d", 2],
    ["Conv2d", 12],
    ["RNN", 24],
    ["Relu"],
]

dataset = pytorch_composer.datasets.RandDataset()
model = pytorch_composer.Model(sequence, dataset)
loop = Loop(model)

code = pytorch_composer.Code([dataset, model, loop])


def check(code, batch_size):
    assert code["batch_size"] == batch_size
    assert code.settings["batch_size"] == batch_size
    assert code[0]["batch_size"] == batch_size
    assert code[0].settings["batch_size"] == batch_size
    assert code[0].variables["x"][0].dim[0] == batch_size
    assert code[1].variables["x"][0].dim[0]  == batch_size
    assert code[2].variables["x"][0].dim[0]  == batch_size
    assert code[2].variables["y"][0].dim  == [batch_size]
    assert code[1].variables["x"][0].dim[1] == code[2].variables["y"][0].vocab.size
    print("asserted " + str(batch_size))
    code()
    print("executed " + str(batch_size))

    
def test_api():
    for _ in range(2):    
        code["batch_size"] = 6
        check(code, 6)
        code.update({"batch_size":8})
        check(code, 8)
        code[0]["batch_size"] = 9
        check(code, 9)
        code[0].update({"batch_size":5})
        check(code, 5)
        code.settings["batch_size"] = 13
        check(code, 13)
        code.settings.update({"batch_size":14})
        check(code, 14)
        code[0].settings["batch_size"] = 3
        check(code, 3)
        code[0].settings.update({"batch_size":17})
        check(code, 17)
        settings = code[0].settings.copy()
        settings["batch_size"] = 15
        code[0].settings = settings
        check(code, 15)
        settings = code.settings.copy()
        settings["batch_size"] = 12
        code.settings = settings
        check(code, 12)

    print("All tests passed")
    
if __name__ == "__main__":
    test_api()
