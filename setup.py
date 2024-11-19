from setuptools import setup

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

if __name__ == "__main__":
    setup(
        name = "SpaDCN",
        version = "1.0.0",
        description = "SpaDCN",
        url = "SpaDCN",
        author = "Xiaosheng Bai",
        author_email = "xs",
        license = "MIT",
        packages = ["SpaDCN"],
        install_requires = ["requests"],
        zip_safe = False,
        include_package_data = True,
        long_description = """ Long Description """,
        long_description_content_type="text/markdown",
    )
