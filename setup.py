import os
import sys
from setuptools import setup, Extension, find_packages

VENDOR_DIR = 'vendor'
MBELIB_ROOT = os.path.join(VENDOR_DIR, 'mbelib-neo')
MBELIB_SRC_DIR = os.path.join(MBELIB_ROOT, 'src')
MBELIB_INC_DIR = os.path.join(MBELIB_ROOT, 'include')

def collect_sources_and_includes(base_dir):
    """
    Recursively collect C source files and include directories.
    Returns sources and include_dirs needed for compilation.
    """
    sources = []
    include_dirs = set()

    for root, _, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            
            if file.endswith('.c'):
                if (file.startswith('test') or 
                    file.endswith('_test.c') or 
                    file.endswith('_dump.c') or 
                    file == 'main.c'):
                    continue
                sources.append(full_path)
            
            elif file.endswith('.h'):
                include_dirs.add(root)

    return sources, list(include_dirs)

def get_extensions():
    if not os.path.exists(MBELIB_SRC_DIR):
        sys.stderr.write(f"Error: Source directory not found at {MBELIB_SRC_DIR}\n")
        sys.exit(1)

    mbelib_sources, internal_includes = collect_sources_and_includes(MBELIB_SRC_DIR)

    if not mbelib_sources:
        sys.stderr.write("Error: No valid source files found in mbelib-neo.\n")
        sys.exit(1)

    wrapper_source = os.path.join('src', 'libambe_wrapper.c')
    all_sources = [wrapper_source] + mbelib_sources

    extra_compile_args = ['-O3', '-fPIC']
    extra_link_args = []
    
    if sys.platform != 'win32':
        extra_compile_args += ['-march=native', '-flto']
        extra_link_args += ['-lm']
    else:
        extra_compile_args = ['/O2']

    final_include_dirs = [MBELIB_INC_DIR] + internal_includes + [MBELIB_ROOT]

    module = Extension(
        name='pyambelib._libambe',
        sources=all_sources,
        include_dirs=final_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [module]

setup(
    name='pyambelib',
    version='0.1.0',
    description='Python wrapper for mbelib-neo',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Your Name',
    license='GPL-2.0-or-later',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=get_extensions(),
    zip_safe=False,
)