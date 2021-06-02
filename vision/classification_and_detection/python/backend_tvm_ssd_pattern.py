import tvm

from tvm import relay
from tvm.relay.dataflow_pattern import *


class Unroller(relay.ExprMutator):
    def Unroll(self, expr, memo):
        self.memo_map = memo
        return self.visit(expr)


class UnrollLoop(DFPatternCallback):
    def __init__(self):
        super().__init__(True)
        self.func_var_ = wildcard()
        self.i_ = wildcard()
        self.max_count_ = wildcard()
        self.cond_ = wildcard()
        self.while_identity_ = wildcard()
        self.output_0_ = wildcard()
        self.output_1_ = wildcard()

        self.equal_ = is_op("equal")(self.cond_, is_constant())
        self.less_ = is_op("less")(self.i_, self.max_count_)
        self.logical_and = is_op("logical_and")(self.equal_, self.less_)

        self.i_increment_ = is_constant()
        self.increment_i_ = self.i_ + self.i_increment_
        self.while_increment_ = is_constant()
        self.increment_while_ = self.while_identity_ + self.while_increment_
        self.cast_ = is_op("cast")(self.increment_while_)
        self.while_limit_ = is_constant()
        self.new_cond_ = is_op("less")(self.cast_, self.while_limit_)

        self.new_output_0_ = wildcard()
        self.new_output_1_ = wildcard()
        self.tuple_0 = TuplePattern([wildcard(), self.new_output_0_])
        self.tuple_1 = TuplePattern([wildcard(), self.new_output_1_])
        self.concat_0 = is_op("concatenate")(self.tuple_0)
        self.concat_1 = is_op("concatenate")(self.tuple_1)

        self.recursion_ = CallPattern(
            self.func_var_,
            [
                self.increment_i_,
                self.max_count_,
                self.new_cond_,
                self.increment_while_,
                self.concat_0,
                self.concat_1,
            ],
        )
        self.tuple_ = TuplePattern(
            [
                self.i_,
                self.max_count_,
                self.cond_,
                self.while_identity_,
                self.output_0_,
                self.output_1_,
            ]
        )
        self.if_ = IfPattern(self.logical_and, self.recursion_, self.tuple_)
        self.func_ = FunctionPattern(
            [
                self.i_,
                self.max_count_,
                self.cond_,
                self.while_identity_,
                self.output_0_,
                self.output_1_,
            ],
            self.if_,
        )
        self.let_ = LetPattern(self.func_var_, self.func_, self.func_var_)

        self.i_init_ = is_constant()
        self.max_count_init_ = is_constant()
        self.cond_init_ = is_constant()
        self.while_identity_init_ = is_constant()
        self.output_0_init_ = wildcard()
        self.output_1_init_ = wildcard()
        self.call_ = CallPattern(
            self.let_,
            [
                self.i_init_,
                self.max_count_init_,
                self.cond_init_,
                self.while_identity_init_,
                self.output_0_init_,
                self.output_1_init_,
            ],
        )
        self.pattern = self.call_

    def callback(self, pre, post, node_map):
        def get_value(pattern):
            return node_map[pattern][0].data.asnumpy().item()

        i_init = get_value(self.i_init_)
        i_increment = get_value(self.i_increment_)
        i_limit = get_value(self.max_count_init_)
        while_limit = get_value(self.while_limit_)
        while_increment = get_value(self.while_increment_)
        while_init = get_value(self.while_identity_init_)
        if (while_limit - while_init) / while_increment == 1 or (
            i_limit - i_init
        ) / i_increment == 1:
            new_vars = {}
            new_vars[node_map[self.i_][0]] = node_map[self.i_init_][0]
            new_vars[node_map[self.max_count_][0]] = node_map[self.max_count_init_][0]
            new_vars[node_map[self.cond_][0]] = node_map[self.cond_init_][0]
            new_vars[node_map[self.while_identity_][0]] = node_map[
                self.while_identity_init_
            ][0]
            new_vars[node_map[self.output_0_][0]] = node_map[self.output_0_init_][0]
            new_vars[node_map[self.output_1_][0]] = node_map[self.output_1_init_][0]
            new_out = relay.Tuple(
                [
                    node_map[self.increment_i_][0],
                    node_map[self.max_count_][0],
                    node_map[self.new_cond_][0],
                    node_map[self.increment_while_][0],
                    node_map[self.new_output_0_][0],
                    node_map[self.new_output_1_][0],
                ]
            )
            new_post = Unroller().Unroll(new_out, new_vars)
            return new_post
        return post


def unroll_loop(expr):
    return rewrite(UnrollLoop(), expr)
