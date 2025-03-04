import lit.formats

config.name = 'KernelFaRer'

config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('opt', "opt -load-pass-plugin ../build/passes/KernelFaRer.so"))
