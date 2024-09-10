import random
import math

#--- COST FUNCTION ------------------------------------------------------------+

# optimize edeceğimiz fonksiyon (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2   #### 50 50 50 45.83 57.23 54.01
    return total
    

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # mevcut uygunluğu (fitness) değerlendir
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # yeni parçacık hızını güncelle
    def update_velocity(self,pos_best_g):
        w=0.5       # atalet ağırlığı
        c1=1        # constant
        c2=2        # constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # parçacık konumunu yeni hız güncellemelerine göre güncelle
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # maximum konum ayarla
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # minimum konum ayarla
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   # grup için en iyi hata
        pos_best_g=[]                   # grup için en iyi konum

        # Sürü oluştur
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # optimizasyon döngüsüne başla
        i=0
        while i < maxiter:
            # Uygunluğu (fitness) değerlendir
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # mevcut parçacığın en iyisi olup olmadığını belirle
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # hızları ve konumları güncelle
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # Sonuç
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)

if __name__ == "__PSO__":
    main()
    
    
initial=[5,5]               # Başlangıç konumları
bounds=[(-sin(5),sin(5)),(-sin(12),sin(12))]  
# başlangıç aralıkları [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1,initial,bounds,num_particles=15,maxiter=30)
