import ast
import pathlib
import sys

TARGET_FUNCS = {"build_applier", "build_algo"}
BASE_DIR = pathlib.Path("microblocks")

class ReturnTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.modified = False
        self.in_target_func_stack = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Track whether we're inside a target function
        self.in_target_func_stack.append(node.name in TARGET_FUNCS)
        self.generic_visit(node)
        self.in_target_func_stack.pop()
        return node

    def visit_Return(self, node: ast.Return):
        # Only transform inside target functions
        if not any(self.in_target_func_stack):
            return node

        # Only transform 4-tuple returns
        if isinstance(node.value, ast.Tuple) and len(node.value.elts) == 4:
            # Replace tuple with BuildResult(...)
            new_call = ast.Call(
                func=ast.Name(id="BuildResult", ctx=ast.Load()),
                args=node.value.elts,  # preserve original expressions
                keywords=[]
            )
            self.modified = True
            return ast.copy_location(ast.Return(value=new_call), node)

        return node


class ImportInjector(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.has_import = False

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Detect existing import of BuildResult from microblocks.base
        if node.module == "microblocks.base":
            for alias in node.names:
                if alias.name == "BuildResult":
                    self.has_import = True
        return node

    def visit_Module(self, node: ast.Module):
        # First pass to detect existing import
        for stmt in node.body:
            self.visit(stmt)

        if not self.has_import:
            # Inject import at top (after docstring if present)
            import_node = ast.ImportFrom(
                module="microblocks.base",
                names=[ast.alias(name="BuildResult", asname=None)],
                level=0
            )
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                node.body.insert(1, import_node)
            else:
                node.body.insert(0, import_node)

        return node


def refactor_file(pyfile: pathlib.Path, dry_run: bool = False):
    try:
        src = pyfile.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[SKIP] {pyfile} (read error: {e})")
        return

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[SKIP] {pyfile} (syntax error: {e})")
        return

    # Transform returns
    rt = ReturnTransformer()
    new_tree = rt.visit(tree)
    ast.fix_missing_locations(new_tree)

    if not rt.modified:
        # No changes needed
        return

    # Ensure import exists
    ii = ImportInjector()
    new_tree = ii.visit(new_tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_src = ast.unparse(new_tree)
    except Exception as e:
        print(f"[ERROR] {pyfile} (unparse failed: {e})")
        return

    if dry_run:
        print(f"[DRY] Would refactor: {pyfile}")
        return

    try:
        pyfile.write_text(new_src, encoding="utf-8")
        print(f"[OK] Refactored: {pyfile}")
    except Exception as e:
        print(f"[ERROR] {pyfile} (write failed: {e})")


def main():
    dry_run = "--dry-run" in sys.argv
    root = BASE_DIR if len(sys.argv) < 2 else pathlib.Path(sys.argv[1])

    if not root.exists():
        print(f"[ERROR] Path not found: {root}")
        sys.exit(1)

    for pyfile in root.rglob("*.py"):
        refactor_file(pyfile, dry_run=dry_run)


if __name__ == "__main__":
    main()
