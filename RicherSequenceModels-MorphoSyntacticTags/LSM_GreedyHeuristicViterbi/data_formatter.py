
def format_data(ip_file, op_file):
    ip_ptr      = open(ip_file, "r")
    op_ptr      = open(op_file, "w")
    counter     = 0
    line        = ip_ptr.readline()
    prev_w_no   = 0
    while line:
        counter += 1
        data_elts   = [elt.strip() for elt in line.strip().split('\t', 3)]
        elt_id      = data_elts[0]
        w_no        = int(elt_id.rsplit('.', 1)[1])
        tag         = data_elts[2]
        if tag.startswith('#'):
            tag = tag.split('#', 1)[1]
        if tag in ["PUN"]:
            main_tag = tag
        else:
            main_tag = tag[0]
        o_line      = "%s\\%s" % (data_elts[1], main_tag)
        if prev_w_no + 1 == w_no:
            op_ptr.write("\t%s" % o_line)
        else:
            op_ptr.write("\n%s" % o_line)
        prev_w_no = w_no

        if counter % 20000 == 0:
            print "processed %d entries" % counter

        line = ip_ptr.readline()
    ip_ptr.close()
    op_ptr.close()




def main():
    ip_file = "data/orwell-en.txt"
    op_file = "data/orwell-en.txt_formatted_main_tag"
    format_data(ip_file, op_file)

if __name__ == '__main__':
    main()
