require 'csv'
require 'libsvm'

x_data = []
y_data = []
# Load data from CSV file into two arrays - one for independent variables X and one for the dependent variable Y
CSV.foreach("./data/admission.csv", :headers => false) do |row|
  x_data.push( [row[0].to_f, row[1].to_f] )
  y_data.push( row[2].to_i )
end

# Divide data into a training set and test set
test_size_percentange = 20.0 # 20.0%
test_set_size = x_data.size * (test_size_percentange/100.to_f)

test_x_data = x_data[0 .. (test_set_size-1)]
test_y_data = y_data[0 .. (test_set_size-1)]

training_x_data = x_data[test_set_size .. x_data.size]
training_y_data = y_data[test_set_size .. y_data.size]


# Convert into proper feature arrays for Libsvm
test_x_data = test_x_data.map {|feature_row| Libsvm::Node.features(feature_row) }
training_x_data = training_x_data.map {|feature_row| Libsvm::Node.features(feature_row) }

# Define our problem using the training dat
problem = Libsvm::Problem.new
problem.set_examples(training_y_data, training_x_data)

## Lets try to find the best C and sigma values
posible_values = [0.0001, 0.0005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]

best_c, best_sigma, best_accuracy = 0,0,0

posible_values.each do |test_c|
  posible_values.each do |test_gamma|
    parameter = Libsvm::SvmParameter.new
    parameter.cache_size = 1 # in megabytes
    parameter.eps = 0.001
    parameter.gamma = test_gamma
    parameter.c = test_c
    parameter.kernel_type = Libsvm::KernelType::RBF

    # Train our model
    model = Libsvm::Model.train(problem, parameter)

    predicted = []
    test_x_data.each do |params|
      predicted.push( model.predict(params) )
    end

    correct = predicted.collect.with_index { |e,i| (e == test_y_data[i]) ? 1 : 0 }.inject{ |sum,e| sum+e }

    accuracy = ((correct.to_f / test_set_size) * 100).round(2)

    if( accuracy > best_accuracy)
      best_accuracy = accuracy
      best_c = test_c
      test_gamma = test_gamma

      puts "New best! Classification Accuracy: #{accuracy}% - C=#{test_c}, gamma=#{test_gamma}"
    end

  end
end
