import ast
import pathlib
import sys

TARGET_FUNCS = {"build_applier", "build_algo", "build_coordinator"}
BASE_DIR = pathlib.Path("microblocks")

class AppendInputTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.in_target_func_stack = []
        self.modified = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.in_target_func_stack.append(node.name in TARGET_FUNCS)
        self.generic_visit(node)
        self.in_target_func_stack.pop()
        return node

    def visit_Return(self, node: ast.Return):
        if not any(self.in_target_func_stack):
            return node

        # Look for return BuildResult(...)
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "BuildResult":
                # Wrap with .appendInput(f"{prev_stages[0]}.applier")
                new_call = ast.Call(
                    func=ast.Attribute(
                        value=node.value,
                        attr="appendInput",
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.JoinedStr([
                            ast.FormattedValue(
                                value=ast.Subscript(
                                    value=ast.Name(id="prev_stages", ctx=ast.Load()),
                                    slice=ast.Constant(value=0),
                                    ctx=ast.Load()
                                ),
                                conversion=-1
                            ),
                            ast.Constant(value=".applier")
                        ])
                    ],
                    keywords=[]
                )
                self.modified = True
                return ast.copy_location(ast.Return(value=new_call), node)

        return node


def refactor_file(pyfile: pathlib.Path, dry_run=False):
    src = pyfile.read_text(encoding="utf-8")
    tree = ast.parse(src)

    transformer = AppendInputTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    if not transformer.modified:
        return

    new_src = ast.unparse(new_tree)

    if dry_run:
        print(f"[DRY] Would refactor {pyfile}")
    else:
        pyfile.write_text(new_src, encoding="utf-8")
        print(f"[OK] Refactored {pyfile}")


def main():
    dry_run = "--dry-run" in sys.argv
    root = BASE_DIR if len(sys.argv) < 2 else pathlib.Path(sys.argv[1])
    for pyfile in root.rglob("*.py"):
        refactor_file(pyfile, dry_run=dry_run)

if __name__ == "__main__":
    main()
