#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "util.h"
#include "matrix.h"
#include "stex.h"

#define BUFF_SIZE 1024
#define HEADER_SIZE 41

size_t objc;
size_t clustc;
size_t max_iter;
double epsilon;
st_matrix dmatrix;
st_matrix memb;

void init_memb() {
    size_t i;
    size_t k;
    double sum;
    double val;
    for(i = 0; i < objc; ++i) {
        sum = 0.0;
        for(k = 0; k < clustc; ++k) {
            val = rand();
            sum += val;
            set(&memb, i, k, val);
        }
        for(k = 0; k < clustc; ++k) {
            set(&memb, i, k, get(&memb, i, k) / sum);
        }
    }
}

void print_memb() {
	printf("Membership:\n");
	size_t i;
	size_t k;
	double sum;
    double val;
	for(i = 0; i < objc; ++i) {
		printf("%u: ", i);
		sum = 0.0;
		for(k = 0; k < clustc; ++k) {
            val = get(&memb, i, k);
			printf("%lf ", val);
			sum += val;
		}
		printf("[%lf]", sum);
		if(!deq(sum, 1.0)) {
			printf("*\n");
		} else {
			printf("\n");
		}
	}
}

double adequacy() {
    size_t h;
    size_t i;
    size_t k;
    double sum_num;
    double sum_den;
    double adeq = 0.0;
    for(k = 0; k < clustc; ++k) {
        sum_num = 0.0;
        sum_den = 0.0;
        for(i = 0; i < objc; ++i) {
            for(h = 0; h < objc; ++h) {
                sum_num += pow(get(&memb, i, k), 2.0) *
                    pow(get(&memb, h, k), 2.0) * get(&dmatrix, i, h);
            }
            sum_den += pow(get(&memb, i, k), 2.0);
        }
        adeq += (sum_num / (2.0 * sum_den));
    }
    return adeq;
}

double run() {
    init_memb();
    print_memb();
    size_t e;
    size_t h;
    size_t i;
    size_t k;
    size_t iter = 1;
    double a_val[clustc];
    double adeq;
    double prev_adeq;
    double adeq_diff;
    st_matrix prev_memb;
    init_st_matrix(&prev_memb, objc, clustc);
    adeq = adequacy();
    printf("Adequacy: %.15lf\n", adeq);
    do {
        printf("Iteration %d:\n", iter);
        mtxcpy(&prev_memb, &memb);
        for(i = 0; i < objc; ++i) {
            double sum_a_val = 0.0;
            for(k = 0; k < clustc; ++k) {
                double term1 = 0.0;
                double term2 = 0.0;
                double sum_den = 0.0;
                for(h = 0; h < objc; ++h) {
                    term1 += pow(get(&memb, h, k), 2.0) *
                        get(&dmatrix, i, h);
                    sum_den += pow(get(&memb, h, k), 2.0);
                    for(e = 0; e < objc; ++e) {
                        term2 += pow(get(&memb, h, k), 2.0) *
                            pow(get(&memb, e, k), 2.0) *
                            get(&dmatrix, h, e);
                    }
                }
                term1 = (2.0 * term1) / sum_den;
                term2 = term2 / pow(sum_den, 2.0);
                a_val[k] = 1.0 / (term1 - term2);
                sum_a_val += a_val[k];
            }
            double v_pos_sum = 0.0;
            double new_memb[clustc];
            for(k = 0; k < clustc; ++k) {
                new_memb[k] = a_val[k] / sum_a_val;
                if(dgt(new_memb[k], 0.0)) {
                    v_pos_sum += a_val[k];
                } else {
                    set(&memb, i, k, 0.0);
                }
            }
            for(k = 0; k < clustc; ++k) {
                if(dgt(new_memb[k], 0.0)) {
                    set(&memb, i, k, a_val[k] / v_pos_sum);
                }
            }
        }
        print_memb();
        prev_adeq = adeq;
        adeq = adequacy();
        printf("Adequacy: %.15lf\n", adeq);
        adeq_diff = prev_adeq - adeq;
        if(adeq_diff < 0.0) {
            adeq_diff = abs(adeq_diff);
            printf("Warn: previous adequacy is greater than "
                   "current (%.15lf).\n", adeq_diff);
        }
        if(adeq_diff < epsilon) {
            break;
        }
    } while (++iter <= max_iter);
    free_st_matrix(&prev_memb);
    return adeq;
}

int main(int argc, char **argv) {
    bool mean_idx = false;
    FILE *cfgfile = fopen(argv[1], "r");
    if(!cfgfile) {
        printf("Error: could not open config file.\n");
    }
    fscanf(cfgfile, "%d", &objc);
    if(objc <= 0) {
        printf("Error: objc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    // reading labels
    int classc;
    int labels[objc];
    fscanf(cfgfile, "%d", &classc);
    size_t i;
    for(i = 0; i < objc; ++i) {
        fscanf(cfgfile, "%d", &labels[i]);
    }
    // reading labels end
    char filename[BUFF_SIZE];
    fscanf(cfgfile, "%s", filename);
    char outfilename[BUFF_SIZE];
    fscanf(cfgfile, "%s", outfilename);
    fscanf(cfgfile, "%d", &clustc);
    if(clustc <= 0) {
        printf("Error: clustc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    int insts;
    fscanf(cfgfile, "%d", &insts);
    if(insts <= 0) {
        printf("Error: insts <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%d", &max_iter);
    if(insts <= 0) {
        printf("Error: max_iter <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%lf", &epsilon);
    if(dlt(epsilon, 0.0)) {
        printf("Error: epsilon < 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fclose(cfgfile);
    freopen(outfilename, "w", stdout);
    printf("###Configuration summary:###\n");
    printf("Number of objects: %d\n", objc);
    printf("Number of clusters: %d\n", clustc);
    printf("Number of instances: %d\n", insts);
    printf("Maximum interations: %d\n", max_iter);
    printf("Epsilon: %.15lf\n", epsilon);
    printf("############################\n");
    st_matrix best_memb;
    // memory allocation start
    init_st_matrix(&dmatrix, objc, objc);
    init_st_matrix(&memb, objc, clustc);
    init_st_matrix(&best_memb, objc, clustc);
    // memory allocation end
    if(!load_data(filename, &dmatrix)) {
        printf("Error: could not load matrix file.\n");
        goto END;
    }
    double avg_partcoef;
    double avg_modpcoef;
    double avg_partent;
    silhouet *csil;
    silhouet *fsil;
    silhouet *avg_csil;
    silhouet *avg_fsil;
    int *pred;
    st_matrix *groups;
    srand(time(NULL));
    size_t best_inst;
    double best_inst_adeq;
    double cur_inst_adeq;
    for(i = 1; i <= insts; ++i) {
        printf("Instance %d:\n", i);
        cur_inst_adeq = run();
        if(mean_idx) {
            pred = defuz(&memb);
            groups = asgroups(pred, objc, classc);
            csil = crispsil(groups, &dmatrix);
            fsil = fuzzysil(csil, groups, &memb, 1.6);
            if(i == 1) {
                avg_partcoef = partcoef(&memb);
                avg_modpcoef = modpcoef(&memb);
                avg_partent = partent(&memb);
                avg_csil = csil;
                avg_fsil = fsil;
            } else {
                avg_partcoef = (avg_partcoef + partcoef(&memb)) / 2.0;
                avg_modpcoef = (avg_modpcoef + modpcoef(&memb)) / 2.0;
                avg_partent = (avg_partent + partent(&memb)) / 2.0;
                avg_silhouet(avg_csil, csil);
                avg_silhouet(avg_fsil, fsil);
                free_silhouet(csil);
                free(csil);
                free_silhouet(fsil);
                free(fsil);
            }
            free(pred);
            free_st_matrix(groups);
            free(groups);
        }
        if(i == 1 || cur_inst_adeq < best_inst_adeq) {
            mtxcpy(&best_memb, &memb);
            best_inst_adeq = cur_inst_adeq;
            best_inst = i;
        }
    }
	printf("\n");
    printf("Best adequacy %.15lf on instance %d.\n", best_inst_adeq,
            best_inst);
    printf("\n");
    print_memb(&best_memb);

    pred = defuz(&best_memb);
    groups = asgroups(pred, objc, classc);
    print_header("Partitions", HEADER_SIZE);
    print_groups(groups);

    if(mean_idx) {
        print_header("Average indexes", HEADER_SIZE);
        printf("\nPartition coefficient: %.10lf\n", avg_partcoef);
        printf("Modified partition coefficient: %.10lf\n", avg_modpcoef);
        printf("Partition entropy: %.10lf (max: %.10lf)\n", avg_partent,
                log(clustc));
    }

    print_header("Best instance indexes", HEADER_SIZE);
    printf("\nPartition coefficient: %.10lf\n", partcoef(&best_memb));
    printf("Modified partition coefficient: %.10lf\n",
            modpcoef(&best_memb));
    printf("Partition entropy: %.10lf (max: %.10lf)\n",
            partent(&best_memb), log(clustc));

    if(mean_idx) {
        print_header("Averaged crisp silhouette", HEADER_SIZE);
        print_silhouet(avg_csil);
        print_header("Averaged fuzzy silhouette", HEADER_SIZE);
        print_silhouet(avg_fsil);
    }

    csil = crispsil(groups, &dmatrix);
    print_header("Best instance crisp silhouette", HEADER_SIZE);
    print_silhouet(csil);
    fsil = fuzzysil(csil, groups, &best_memb, 1.6);
    print_header("Best instance fuzzy silhouette", HEADER_SIZE);
    print_silhouet(fsil);

    if(mean_idx) {
        free_silhouet(avg_csil);
        free(avg_csil);
        free_silhouet(avg_fsil);
        free(avg_fsil);
    }
    free_silhouet(csil);
    free(csil);
    free_silhouet(fsil);
    free(fsil);
    free(pred);
    free_st_matrix(groups);
    free(groups);
END:
    fclose(stdout);
    free_st_matrix(&dmatrix);
    free_st_matrix(&memb);
    free_st_matrix(&best_memb);
    return 0;
}
