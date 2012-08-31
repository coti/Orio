import orio.module.loops.ast as ast
import orio.main.util.globals as g

#----------------------------------------------------------------------------------------------------------------------
class CodeGen:
    '''The code generator for the AST classes'''

    def __init__(self, language='C'):
        '''Instantiates a code generator'''

        self.generator = None
        if language.lower() in ['c', 'c++', 'cxx']:
            self.generator = CodeGen_C()
        #elif language.lower() in ['f', 'f90', 'fortran']:
        #    self.generator = CodeGen_F()
        #elif language.lower() in ['cuda']:
        #    from orio.module.loop.codegen_cuda import CodeGen_CUDA
        #    self.generator = CodeGen_CUDA()
        else:
            g.err('orio.module.loops.printer: Unknown language specified for code generation: %s' % language)
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  '):
        '''Generates code that corresponds to the given AST'''

        return self.generator.generate(tnode, indent, extra_indent)


class CodeGen_C(CodeGen):
    '''The code generator for the AST classes'''

    def __init__(self):
        '''To instantiate a code generator'''
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  '):
        '''To generate code that corresponds to the given AST'''

        s = ''

        if isinstance(tnode, ast.Comment):
            s += indent + '/*' + tnode.text + '*/\n'
            
        elif isinstance(tnode, ast.LitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ast.ArrayRefExp):
            s += self.generate(tnode.exp, indent, extra_indent)
            s += '[' + self.generate(tnode.sub, indent, extra_indent) + ']'

        elif isinstance(tnode, ast.CallExp):
            s += self.generate(tnode.exp, indent, extra_indent) + '('
            s += ','.join(map(lambda x: self.generate(x, indent, extra_indent), tnode.args))
            s += ')'

        elif isinstance(tnode, ast.UnaryExp):
            s += self.generate(tnode.exp, indent, extra_indent)
            if tnode.op_type == tnode.PLUS:
                s = '+' + s
            elif tnode.op_type == tnode.MINUS:
                s = '-' + s
            elif tnode.op_type == tnode.LNOT:
                s = '!' + s
            elif tnode.op_type == tnode.PRE_INC:
                s = ' ++' + s
            elif tnode.op_type == tnode.PRE_DEC:
                s = ' --' + s
            elif tnode.op_type == tnode.POST_INC:
                s += '++ '
            elif tnode.op_type == tnode.POST_DEC:
                s += '-- '
            elif tnode.op_type == tnode.DEREF:
                s = '*' + s
            elif tnode.op_type == tnode.ADDRESSOF:
                s = '&' + s
            else:
                g.err('orio.module.loop.codegen internal error: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
            s += self.generate(tnode.lhs, indent, extra_indent)
            if tnode.op_type == tnode.PLUS:
                s += '+'
            elif tnode.op_type == tnode.MINUS:
                s += '-'
            elif tnode.op_type == tnode.MUL:
                s += '*'
            elif tnode.op_type == tnode.DIV:
                s += '/'
            elif tnode.op_type == tnode.MOD:
                s += '%'
            elif tnode.op_type == tnode.LT:
                s += '<'
            elif tnode.op_type == tnode.GT:
                s += '>'
            elif tnode.op_type == tnode.LE:
                s += '<='
            elif tnode.op_type == tnode.GE:
                s += '>='
            elif tnode.op_type == tnode.EE:
                s += '=='
            elif tnode.op_type == tnode.NE:
                s += '!='
            elif tnode.op_type == tnode.LOR:
                s += '||'
            elif tnode.op_type == tnode.LAND:
                s += '&&'
            elif tnode.op_type == tnode.EQ:
                s += '='
            elif tnode.op_type == tnode.EQPLUS:
                s += '+='
            elif tnode.op_type == tnode.EQMINUS:
                s += '-='
            elif tnode.op_type == tnode.EQMULT:
                s += '*='
            elif tnode.op_type == tnode.EQDIV:
                s += '/='
            elif tnode.op_type == tnode.EQMOD:
                s += '%='
            elif tnode.op_type == tnode.COMMA:
                s += ','
            else:
                g.err('orio.module.loops.printer internal error: unknown binary operator type: %s' % tnode.op_type)
            s += self.generate(tnode.rhs, indent, extra_indent)

        elif isinstance(tnode, ast.ParenExp):
            s += '(' + self.generate(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.CompStmt):
            s += indent + '{\n'
            for stmt in tnode.stmts:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, ast.IfStmt):
            s += indent + 'if (' + self.generate(tnode.test, indent, extra_indent) + ') '
            if isinstance(tnode.true_stmt, ast.CompStmt):
                tstmt_s = self.generate(tnode.true_stmt, indent, extra_indent)
                s += tstmt_s[tstmt_s.index('{'):]
                if tnode.false_stmt:
                    s = s[:-1] + ' else '
            else:
                s += '\n'
                s += self.generate(tnode.true_stmt, indent + extra_indent, extra_indent)
                if tnode.false_stmt:
                    s += indent + 'else '
            if tnode.false_stmt:
                if isinstance(tnode.false_stmt, ast.CompStmt):
                    tstmt_s = self.generate(tnode.false_stmt, indent, extra_indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.generate(tnode.false_stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.ForStmt):
            s += indent + 'for ('
            if tnode.init:
                s += self.generate(tnode.init, indent, extra_indent)
            s += '; '
            if tnode.test:
                s += self.generate(tnode.test, indent, extra_indent)
            s += '; '
            if tnode.iter:
                s += self.generate(tnode.iter, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.WhileStmt):
            s += indent + 'while (' + self.generate(tnode.test, indent, extra_indent) + ') '
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.VarDec):
            s += indent + str(tnode.type_name) + ' '
            s += ', '.join(map(lambda x: self.generate(x, indent, extra_indent), tnode.var_inits))
            s += ';\n'

        elif isinstance(tnode, ast.ParamDec):
            s += indent + str(tnode.ty) + ' ' + str(tnode.name)

        elif isinstance(tnode, ast.FunDec):
            s += indent + str(tnode.return_type) + ' ' + str(tnode.modifiers)
            s += tnode.name + '('
            s += ', '.join(map(lambda x: self.generate(x, indent, extra_indent), tnode.params))
            s += ')' + self.generate(tnode.body, indent, extra_indent)

        elif isinstance(tnode, ast.TransformStmt):
            g.err('orio.module.loop.codegen internal error: a transformation statement is never generated as an output')

        else:
            g.err('orio.module.loop.codegen internal error: unrecognized type of AST: %s' % tnode.__class__.__name__)

        return s
#----------------------------------------------------------------------------------------------------------------------

