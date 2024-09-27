from autodiff import Tape, Node, Number


alias width = 4
alias type = DType.float32
fn main():
    var tape = Tape[width, type]()
    var number1 = Number[width, type](4.0, tape)
    var number2 = Number[width, type](6.0, tape)
    tape.set_mark()
    var number3 = number1 * number2
    number3.propagate_adjoints(1)
    print(number1.mean_grad(), number2.mean_grad())
