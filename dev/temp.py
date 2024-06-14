template = """
Host c{0}
    HostName c-00{0}.cs.tau.ac.il
    User roeehendel
    IdentityFile C:\\sshkeys\\taucs
"""

for i in range(1, 9):
    print(template.format(i))